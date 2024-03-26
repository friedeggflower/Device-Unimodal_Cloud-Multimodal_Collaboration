from detectron2.config import get_cfg
import argparse
from bua import add_config
from detectron2.engine import DefaultTrainer, default_setup, launch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import ImageList
from torch.nn import AdaptiveAvgPool2d, Linear, Sequential, ReLU, Dropout, CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch.autograd import Variable

from torchvision.models import resnet18
from collections import OrderedDict
from torch import nn 
import random

import torch.distributed as dist
import time

import cv2
import numpy as np
import torch
import os
import json
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

PIXEL_MEAN = np.array([[[103.5300]],[[116.2800]],[[123.6750]]]).astype(np.float32)
PIXEL_STD = np.array([[[1.]],[[1.]],[[1.]]]).astype(np.float32)

TEST_SCALES = (600,)
TEST_MAX_SIZE = 1000

PIXEL_MEAN = torch.from_numpy(PIXEL_MEAN)
PIXEL_STD = torch.from_numpy(PIXEL_STD)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_config(args, cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    default_setup(cfg, args)
    return cfg

def get_image_blob(im, pixel_means=0.0):
    """Converts an image into a network input.
    Arguments:
        im (ndarray): a color image
    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    pixel_means = np.array([[pixel_means]])
    dataset_dict = {}
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= pixel_means

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    for target_size in TEST_SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > TEST_MAX_SIZE:
            im_scale = float(TEST_MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)

    dataset_dict["image"] = torch.from_numpy(im).permute(2, 0, 1)
    dataset_dict["im_scale"] = im_scale

    return dataset_dict

def load_pth(model, local_resume_path=None):
    if local_resume_path is None:
        local_resume_path = "/mnt/workspace/bottom-up-attention.pytorch-master/checkpoint/backbone/test.pth"
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

def save_to_local(model, save_path=None):
    torch.save(model.state_dict(), "/mnt/workspace/bottom-up-attention.pytorch-master/checkpoint/backbone/test.pth")
    print("saved")

class classifier(torch.nn.Module):
    def __init__(self, backbone,):
        super(classifier, self).__init__()
        self.backbone = backbone
        self.avg_pool = AdaptiveAvgPool2d(output_size = (2,2))
        self.classfier = Sequential(Linear(4096, 1024),
                                    ReLU(),
                                    Dropout(0.2),
                                    Linear(1024, 48))
    
    def forward(self, x):
        x = self.backbone(x)['res4']
        bs = x.shape[0]
        x = self.avg_pool(x)
        x = x.view(bs, -1)
        x = self.classfier(x)
        return x

class vis_Dataset(Dataset):
    def __init__(self, idx_list):
        self.idx_list = idx_list

        meta_path = '/Public_Dataset/Data/meta.json'
        with open(meta_path, "r") as f:
            self.meta = json.load(f)
        f.close()

        cate2label_path = '/Public_Dataset/Data/cate2label.json'
        with open(cate2label_path, 'r') as f:
            self.cate2label = json.load(f)
        f.close()

    def __len__(self):
        return len(self.idx_list)

    def preprocess(self, img):
        im_orig = img.astype(np.float32, copy=True)

        im_shape = im_orig.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        for target_size in TEST_SCALES:
            im_scale = float(target_size) / float(im_size_min)
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > TEST_MAX_SIZE:
                im_scale = float(TEST_MAX_SIZE) / float(im_size_max)
            im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

        im = torch.from_numpy(im).permute(2, 0, 1)
        im = (im - PIXEL_MEAN)/PIXEL_STD
        return im

    def __getitem__(self, i):
        idx = self.idx_list[i]
        img_path = os.path.join('/Public_Dataset/Data',self.meta[str(idx)]['img_path'])
        cate = self.meta[str(idx)]['cate']
        label = self.cate2label[cate]

        img = cv2.imread(img_path)
        # print("img", img)
        img = self.preprocess(img)

        return img, label, idx

def my_collate_fn(inputs):
    batch_size = len(inputs)
    imgs = []
    labels = []
    idxs = []
    for data in inputs:
        imgs.append(data[0])
        labels.append(data[1])
        idxs.append(data[2])
    imgs = torch.stack(imgs, dim=0)
    # imgs = ImageList.from_tensors(imgs,0)
    labels = torch.from_numpy(np.array(labels))
    return imgs, labels, idxs

def build_dataloader(dataset, collate_fn, is_train, cfg):
    batch_size = cfg.batch_size
        
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            drop_last=False,
                            drop_last = is_train,
                            shuffle = is_train,
                            num_workers=cfg.n_workers,
                            pin_memory=True, collate_fn=collate_fn,)
    return dataloader


def test(model, test_loader, cfg, args):
    total = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader, total=len(test_loader)):
        # for data in test_dataloader:
            imgs, labels, idxs = data
            imgs = Variable(imgs.cuda())
            labels = Variable(labels.cuda())

            outputs = model(imgs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            _correct =  (predicted == labels).sum().detach().cpu()
            correct += _correct

    correct = int(correct)
    total = int(total)

    print("test acc", correct,"/", total, ':', correct/total)
    return correct, total, round(correct/total,4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Object Detection2 Inference")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--nproc_per_node", type=int, default=2)

    parser.add_argument("--config_file",default='configs/d2/test-d2-r50.yaml',metavar="FILE",)
    parser.add_argument("opts",help="Modify config options using the command-line",default=None,nargs=argparse.REMAINDER,)
    parser.add_argument("--mode", default="d2", type=str)
    parser.add_argument("--resume", action="store_true") # default=False
    args = parser.parse_args()
    cfg = setup(args)


    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
    model.eval()

    with open('/Public_Dataset/Data/idx/user2vistrainidx.json', 'r') as f:
        user2vistrainidx = json.load(f)
    f.close()

    with open('/Public_Dataset/Data/idx/after1_user2testidx.json', 'r') as f:
        user2testidx = json.load(f)
    f.close()

    users_list = sorted(list(user2vistrainidx.keys()))
    print(users_list)
    users_list = users_list[args.local_rank*5:(args.local_rank+1)*5]

    cfg.batch_size = 32 
    cfg.n_workers = 1
    cfg.epochs = 10

    cfg.lr = 0.0001
    cfg.step_size = 50000 
    cfg.gamma = 0.99
    cfg.resume = False

    for user in users_list:
        print("user: %s ==========================Train==========================" % (user))

        save_root = '/Public_Dataset/Device_unimodal/checkpoint/users/lr{}_step{}_gamma{}_adam'.format(cfg.lr, cfg.step_size, cfg.gamma)
        if not os.path.exists(save_root):
            os.mkdir(save_root)
        save_root = os.path.join(save_root, user)
        if not os.path.exists(save_root):
            os.mkdir(save_root)
                
        train_dataset = vis_Dataset(user2vistrainidx[user])
        test_dataset = vis_Dataset(user2testidx[user])
        train_dataloader = build_dataloader(train_dataset, my_collate_fn, True, cfg)
        test_dataloader = build_dataloader(test_dataset, my_collate_fn, False, cfg)

        my_classifier = classifier(backbone=model.backbone)
        local_resume_path = os.path.join('/Public_Dataset/Device_unimodal/checkpoint/backbone/first/lr0.001_step50000_gamma0.99_adam/49_train-38150.pth')
        load_pth(my_classifier, local_resume_path)

        #freeze classifier
        base_ids = list(map(id, my_classifier.classfier.parameters()))
        for para in my_classifier.classfier.parameters():
            para.requires_grad = False
        for para in my_classifier.backbone.parameters():
            para.requires_grad = True
 
        paras = [{'params': filter(lambda p: p.requires_grad, my_classifier.parameters()), 'lr': cfg.lr}]

        optimizer = optim.Adam(paras, lr=cfg.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma, last_epoch=-1)
        my_classifier.cuda(args.local_rank)

        cirterion = CrossEntropyLoss().cuda(args.local_rank)
        global_step = 0
        train_losses = [] 
        lr_list = []
        test_res = {}

        if cfg.resume:
            epochs_list = range(40, cfg.epochs)
            with open(os.path.join(save_root,"test_res.json") ,'r') as f:
                test_res = json.load(f)
            f.close()
        else:
            epochs_list = range(cfg.epochs)

        correct, total, acc = test(my_classifier, test_dataloader, cfg, args)
        test_res['-1'] = str(correct)+'/'+ str(total)+'/'+ str(acc)

        for epoch in epochs_list:
            my_classifier.train()
            running_loss = 0.0
            # if args.local_rank == 0:
            print("Epoch: %d ==========================Train==========================" % (epoch + 1))

            for step, batch_data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
                global_step+=1
                imgs, labels, idxs = batch_data
                imgs = Variable(imgs.to(args.local_rank))
                labels = Variable(labels.to(args.local_rank))
                outputs = my_classifier(imgs)

                optimizer.zero_grad()  
                loss = cirterion(outputs, labels)
                loss.backward()
                optimizer.step() 
                lr_scheduler.step()

                cur_loss = float(loss)

                if not cur_loss > 4:
                    train_losses.append(cur_loss)
                running_loss += cur_loss
                lr_list.append(optimizer.param_groups[0]['lr'])
  
                if step % 50 == 0:
                    print('[%d %5d] loss: %.3f' % (epoch + 1, step, running_loss / 200))

            if True:
                model_path_suffix = "{}_train-{}.pth".format(epoch, global_step)
                model_save_path = os.path.join(save_root, model_path_suffix)

                checkpoint_dict = { 'model_state_dict': my_classifier.state_dict(), 
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
            
            correct, total, acc = test(my_classifier, test_dataloader, cfg, args)
            test_res[epoch] = str(correct)+'/'+ str(total)+'/'+ str(acc)

            if True:
                json_save_path = os.path.join(save_root, 'test_res.json')
                with open(json_save_path, 'w') as f:
                    json.dump(test_res, f)
                f.close()




