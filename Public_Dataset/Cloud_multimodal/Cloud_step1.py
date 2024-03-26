"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UNITER finetuning for Image-Text Retrieval
"""
import argparse
import os
from os.path import exists, join
from time import time
import json
from collections import OrderedDict
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, ConcatDataset
from apex import amp
# from apex.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.nn import CrossEntropyLoss

# from horovod import torch as hvd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time


from data import PrefetchLoader, TxtTokLmdb, DetectFeatLmdb, ClassDataset, class_collate
from model.class_model_122_small import UniterForClassSmall
from optim import get_lr_sched
from optim.misc import build_optimizer

from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.distributed import (all_reduce_and_rescale_tensors, all_gather_list,
                               broadcast_tensors)
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp, parse_with_config, set_dropout, set_random_seed
from utils.const import IMG_DIM
from utils.itm_eval import evaluate


def build_dataloader(dataset, collate_fn, is_train, opts):
    if is_train:
        batch_size = opts.train_batch_size 
    else:
        batch_size = 16
    #for ddp
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=is_train)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, drop_last=is_train,
                            num_workers=opts.n_workers,
                            pin_memory=True, collate_fn=collate_fn,
                            sampler=train_sampler)
    dataloader = PrefetchLoader(dataloader)
    return dataloader

def load_pth(model, local_resume_path):
    checkpoint_state = torch.load(local_resume_path, map_location=lambda storage, loc: storage)['model_state_dict']
    model_state_dict = checkpoint_state

    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        if k.startswith("module."):
            name = k[7:]  # remove `module.`
        else:
            name = k
        if "queue" in name:
            name = name + "_mis_match"
        new_state_dict[name] = v
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    print("missing_keys:\n{}".format(missing_keys))
    print("unexpected_keys:\n{}".format(unexpected_keys))

def main(opts):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(opts.local_rank)

    set_random_seed(opts.seed)

    #dataset
    txt_path = 'Public_Dataset/Data/txt_db_mask_subcate'
    img_path = 'Public_Dataset/Data/img_db_two_stage/nms_thres_0.5'
    txt_db = TxtTokLmdb(txt_path)
    img_db = DetectFeatLmdb(img_path)

    train_cate2idx_path = 'Public_Dataset/Data/idx/cate2idx_1.json'
    test_cate2idx_path = 'Public_Dataset/Data/idx/after1_user2testidx.json'
    train_dataset = ClassDataset(train_cate2idx_path, txt_db, img_db)
    test_dataset = ClassDataset(test_cate2idx_path, txt_db, img_db)

    #main paras
    opts.epochs = 40
    opts.train_batch_size = 32 # if there are 4 devices in ddp
    opts.learning_rate = 3e-5
    opts.layer_num = 3

    train_dataloader = build_dataloader(train_dataset, class_collate, True, opts)
    test_dataloader = build_dataloader(test_dataset, class_collate, False, opts)

    save_root = 'Public_Dataset/Cloud_multimodal/checkpoint/step1/lr{}_adam_{}layer'.format(opts.learning_rate, opts.layer_num)
    if args.local_rank == 0:
        if not os.path.exists(save_root):
            os.mkdir(save_root)

    if opts.checkpoint:
        checkpoint = torch.load(opts.checkpoint)
    else:
        checkpoint = {}

    # from pretrained model
    model = UniterForClassSmall.from_pretrained(opts.model_config, state_dict=checkpoint,img_dim=IMG_DIM, layer_num = opts.layer_num)
    model.cuda(opts.local_rank)

    # make sure every process has same model parameters in the beginning
    # for ddp
    # broadcast_tensors([p.data for p in model.parameters()], 0)
    set_dropout(model, opts.dropout)

    # Prepare optimizer
    optimizer = build_optimizer(model, opts)

    model = nn.parallel.DistributedDataParallel(model, device_ids=[opts.local_rank], find_unused_parameters=True)
    cirterion = CrossEntropyLoss()

    train_losses = []
    lr_list = []
    test_res = {}
    loss_dict = {}

    if os.path.exists(os.path.join(save_root, 'test_res.json')):
        with open(os.path.join(save_root, 'test_res.json'),'r') as f:
            test_res = json.load(f)
        f.close()

    if os.path.exists(os.path.join(save_root, 'loss.json')):
        with open(os.path.join(save_root, 'loss.json'),'r') as f:
            loss_dict = json.load(f)
        f.close()

    if not '-1' in test_res.keys():
        acc = test(model, test_dataloader, opts)
        test_res['-1'] = acc

    for epoch in range(opts.epochs):
        running_loss = 0.0
        if opts.local_rank == 0:
            print("Epoch: %d ==========================Train==========================" % (epoch + 1))
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            if (step+1)%50==0:
                time.sleep(2)
            model.train()
            labels = batch['labels']
            outputs = model(batch)
            loss = cirterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            #记录训练过程中的loss和lr
            all_losses = [torch.zeros_like(loss) for _ in range(torch.cuda.device_count())]
            torch.distributed.all_gather(all_losses, loss)
            all_losses = [float(loss) for loss in all_losses]
            cur_loss = np.mean(np.array(all_losses))

            train_losses.append(cur_loss)
            running_loss += cur_loss
            lr_list.append(optimizer.param_groups[0]['lr'])
            if args.local_rank==0 and step % 400 == 0:
                print('loss: %.3f' % (running_loss / 200))

            torch.cuda.empty_cache()

        loss_dict[epoch] = running_loss

        if args.local_rank == 0:
            model_path_suffix = "{}_train.pth".format(epoch)
            model_save_path = os.path.join(save_root, model_path_suffix)

            checkpoint_dict = { 'model_state_dict': model.state_dict(), 
                                'optim_state_dict': optimizer.state_dict(), }
            torch.save(checkpoint_dict, model_save_path)

            plt.figure()
            img_name = 'loss.jpg' 
            img_save_path = os.path.join(save_root,img_name) 
            plt.subplot(121)
            plt.plot(train_losses, label='Training loss')
            plt.title('loss')
            plt.ylabel('loss')
            plt.xlabel('step')

            plt.subplot(122)
            plt.plot(lr_list, label='lr')
            plt.title('lr')
            plt.ylabel('lr')
            plt.xlabel('step')
            plt.savefig(img_save_path)   

        # test
        acc = test(model, test_dataloader, opts)
        test_res[epoch] = acc
        if args.local_rank==0:
            json_save_path = os.path.join(save_root, 'test_res.json')
            with open(json_save_path, 'w') as f:
                json.dump(test_res, f)
            f.close()
            loss_save_path = os.path.join(save_root, 'loss.json')
            with open(loss_save_path, 'w') as f:
                json.dump(loss_dict, f)
            f.close()

def test(model, test_loader, opts):
    total = 0
    correct = 0
    step = 0
    with torch.no_grad():
        model.eval()
        for batch in tqdm(test_loader,total=len(test_loader)):
            step+=1
            if step%50==0:
                time.sleep(2)
            labels = batch['labels']
            outputs = model.module(batch)
            _, predicted = torch.max(outputs.data, dim=1)


            all_predicted = [torch.zeros_like(predicted) for _ in range(torch.cuda.device_count())]
            torch.distributed.all_gather(all_predicted, predicted)
            all_predicted = torch.cat(all_predicted, dim=0)

            all_labels = [torch.zeros_like(labels) for _ in range(torch.cuda.device_count())]
            torch.distributed.all_gather(all_labels, labels)
            all_labels = torch.cat(all_labels, dim=0)
            
            _correct =  int((all_predicted == all_labels).sum().detach().cpu())
            total += int(all_labels.size(0))
            correct += _correct

    correct = int(correct)
    total = int(total)
    if opts.local_rank == 0:
        print("test acc", correct,"/", total, ':', correct/total)
    return round(correct/total,4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')
    parser.add_argument("--checkpoint",
                        default=None, type=str,
                        help="pretrained MLM")

    # Prepro parameters
    parser.add_argument('--max_txt_len', type=int, default=100,
                        help='max number of tokens in text (BERT BPE)')
    parser.add_argument('--conf_th', type=float, default=0.2,
                        help='threshold for dynamic bounding boxes '
                             '(-1 for fixed)')
    parser.add_argument('--max_bb', type=int, default=100,
                        help='max number of bounding boxes')
    parser.add_argument('--min_bb', type=int, default=10,
                        help='min number of bounding boxes')
    parser.add_argument('--num_bb', type=int, default=36,
                        help='static number of bounding boxes')

    # training parameters
    parser.add_argument("--train_batch_size", default=128, type=int,
                        help="Total batch size for training. "
                             "(batch by examples)")
    parser.add_argument("--negative_size", default=1, type=int,
                        help="Number of negative samples per positive sample")
    parser.add_argument("--inf_minibatch_size", default=400, type=int,
                        help="batch size for running inference. "
                             "(used for validation, and evaluation)")

    parser.add_argument("--margin", default=0.2, type=float,
                        help="margin of ranking loss")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
                        help="Number of updates steps to accumualte before "
                             "performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--optim", default='adamw',
                        choices=['adam', 'adamax', 'adamw'],
                        help="optimizer")
    parser.add_argument("--betas", default=[0.9, 0.98], nargs='+',
                        help="beta for adam optimizer")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="tune dropout regularization")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="weight decay (L2) regularization")
    parser.add_argument("--grad_norm", default=0.25, type=float,
                        help="gradient clipping (-1 for no clipping)")


    # device parameters
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--full_val', action='store_true',
                        help="Always run full evaluation during training")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--n_workers', type=int, default=8,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true',
                        help="pin memory")
    parser.add_argument("--local_rank", type=int, help="")

    # can use config files
    parser.add_argument('--config', default='Public_Dataset/Cloud_multimodal/config/train_class.json',help='JSON config files')

    args = parse_with_config(parser)

    # options safe guard
    if args.conf_th == -1:
        assert args.max_bb + args.max_txt_len + 2 <= 512
    else:
        assert args.num_bb + args.max_txt_len + 2 <= 512

    main(args)
