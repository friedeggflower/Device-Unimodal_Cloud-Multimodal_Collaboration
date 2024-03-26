import argparse
import os
import sys
import torch
# import tqdm
import cv2
import numpy as np
import json
from collections import OrderedDict
from tqdm import tqdm
sys.path.append('detectron2')

from detectron2.structures import Boxes
from typing import List, Tuple, Union

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.structures import Instances
from torch.utils.data import Dataset, DataLoader

from bua.d2 import add_attribute_config
from bua import add_config
# from bua.caffe.modeling.box_regression import BUABoxes
from torch.nn import functional as F
from detectron2.modeling import postprocessing
from bua.d2.modeling.roi_heads import AttributeROIHeads, AttributeRes5ROIHeads,register

# import ray
# from ray.actor import ActorHandle

from glob import glob
TEST_SCALES = (600,)
TEST_MAX_SIZE = 1000

def get_ids():
    idx2path_name = 'Public_Dataset/Data/idx2path.json'
    if os.path.exists(idx2path_name):
        with open(idx2path_name, 'r') as f:
            idx2path = json.load(f)
        f.close()
        ids = sorted(idx2path.keys())
    else:
        ids = []
        idx2path = {}

        train_root = 'Public_Dataset/Data/train/image'
        fn_list = os.listdir(train_root)
        for fn in fn_list:
            _idx = fn.split('.')[0]
            ids.append(_idx)
            idx2path[_idx] = os.path.join(train_root, fn)
        print("train", len(ids))
        
        test_root = 'Public_Dataset/Data/valid/image'
        fn_list = os.listdir(test_root)
        for fn in fn_list:
            _idx = fn.split('.')[0]
            ids.append(_idx)
            idx2path[_idx] = os.path.join(test_root, fn)

        with open(idx2path_name, 'w') as f:
            json.dump(idx2path, f)
        f.close()
    return ids, idx2path

def switch_extract_mode(mode):
    if mode == 'roi_feats':
        switch_cmd = ['MODEL.BUA.EXTRACTOR.MODE', 1]
    elif mode == 'bboxes':
        switch_cmd = ['MODEL.BUA.EXTRACTOR.MODE', 2]
    elif mode == 'bbox_feats':
        switch_cmd = ['MODEL.BUA.EXTRACTOR.MODE', 3, 'MODEL.PROPOSAL_GENERATOR.NAME', 'PrecomputedProposals']
    else:
        print('Wrong extract mode! ')
        exit()
    return switch_cmd

class BUABoxes(Boxes):
    """
        This structure stores a list of boxes as a Nx4 torch.Tensor.
        It supports some common methods about boxes
        (`area`, `clip`, `nonempty`, etc),
        and also behaves like a Tensor
        (support indexing, `to(device)`, `.device`, and iteration over all boxes)

        Attributes:
            tensor: float matrix of Nx4.
        """

    BoxSizeType = Union[List[int], Tuple[int, int]]
    def __init__(self, tensor: torch.Tensor):
        super().__init__(tensor)

    def clip(self, box_size: BoxSizeType) -> None:
        """
        NOTE: In order to be the same as bottom-up-attention network, we have
        defined the new clip function.

        Clip (in place) the boxes by limiting x coordinates to the range [0, width]
        and y coordinates to the range [0, height].

        Args:
            box_size (height, width): The clipping box's size.
        """
        assert torch.isfinite(self.tensor).all(), "Box tensor contains infinite or NaN!"
        TO_REMOVE = 1
        h, w = box_size
        self.tensor[:, 0].clamp_(min=0, max=w - TO_REMOVE)
        self.tensor[:, 1].clamp_(min=0, max=h - TO_REMOVE)
        self.tensor[:, 2].clamp_(min=0, max=w - TO_REMOVE)
        self.tensor[:, 3].clamp_(min=0, max=h - TO_REMOVE)

    def nonempty(self, threshold: int = 0) -> torch.Tensor:
        """
        NOTE: In order to be the same as bottom-up-attention network, we have
        defined the new nonempty function.

        Find boxes that are non-empty.
        A box is considered empty, if either of its side is no larger than threshold.

        Returns:
            Tensor:
                a binary vector which represents whether each box is empty
                (False) or non-empty (True).
        """
        TO_REMOVE = 1
        box = self.tensor
        widths = box[:, 2] - box[:, 0] + TO_REMOVE
        heights = box[:, 3] - box[:, 1] + TO_REMOVE
        keep = (widths > threshold) & (heights > threshold)
        return keep

    def filter_boxes(self):
        box = self.tensor
        keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
        return keep

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Boxes":
        """
        Returns:
            BUABoxes: Create a new :class:`BUABoxes` by indexing.

        The following usage are allowed:
        1. `new_boxes = boxes[3]`: return a `Boxes` which contains only one box.
        2. `new_boxes = boxes[2:10]`: return a slice of boxes.
        3. `new_boxes = boxes[vector]`, where vector is a torch.BoolTensor
           with `length = len(boxes)`. Nonzero elements in the vector will be selected.

        Note that the returned Boxes might share storage with this Boxes,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return BUABoxes(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert b.dim() == 2, "Indexing on Boxes with {} failed to return a matrix!".format(item)
        return BUABoxes(b)

def get_image_blob(im, pixel_means=0.0):
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

def build_dataloader(dataset, collate_fn, is_train, cfg):
    batch_size = cfg.batch_size
    #for ddp
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, drop_last=False,
                            num_workers=cfg.n_workers,
                            pin_memory=True, collate_fn=collate_fn,)
    return dataloader

def setup(args):
    cfg = get_cfg()
    add_attribute_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.merge_from_list(switch_extract_mode(args.extract_mode))
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection2 Inference")
    parser.add_argument(
        "--config-file",
        default="./configs/d2/test-d2-r50.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument('--num-cpus', default=1, type=int, 
                        help='number of cpus to use for ray, 0 means no limit')

    parser.add_argument('--gpus', dest='gpu_id', help='GPU id(s) to use',
                        default='0,1', type=str)

    parser.add_argument("--mode", default="d2", type=str, help="bua_caffe, ...")

    parser.add_argument('--extract-mode', default='roi_feats', type=str,
                        help="'roi_feats', 'bboxes' and 'bbox_feats' indicates \
                        'extract roi features directly', 'extract bboxes only' and \
                        'extract roi features with pre-computed bboxes' respectively")

    parser.add_argument('--min-max-boxes', default='min_max_default', type=str, 
                        help='the number of min-max boxes of extractor')

    parser.add_argument('--out-dir', dest='output_dir',
                        help='output directory for features',
                        default="features")
    parser.add_argument('--image-dir', dest='image_dir',
                        help='directory with images',
                        default="image")
    parser.add_argument('--bbox-dir', dest='bbox_dir',
                        help='directory with bbox',
                        default="bbox")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--id", type=int, default=0)
    return parser.parse_args()

def model_inference(model, batched_inputs, extract_mode, dump_folder, image_h, image_w, attribute_on=False):
    images = model.preprocess_image(batched_inputs)
    features = model.backbone(images.tensor)
    
    if extract_mode != 3:
        proposals, _ = model.proposal_generator(images, features, None)
    else: # feats_by_box
        assert "proposals" in batched_inputs[0]
        print("proposals")
        proposals = [x["proposals"].to(model.device) for x in batched_inputs]
    _, pooled_features, _ = model.roi_heads.get_roi_features(features, proposals)  # fc7 feats
    predictions = model.roi_heads.box_predictor(pooled_features)
    cls_lables = torch.argmax(predictions[0], dim=1)
    
    cls_probs = F.softmax(predictions[0], dim=-1)
    cls_probs = cls_probs[:, :-1]  # background is last
    if extract_mode != 3:
        predictions, r_indices = model.roi_heads.box_predictor.inference(predictions, proposals)

        if attribute_on:
            attr_scores = model.roi_heads.forward_attribute_score(pooled_features, cls_lables)
            attr_probs = F.softmax(attr_scores, dim=-1)
            attr_probs = attr_probs[r_indices]

        # postprocess
        height = images[0].shape[1]
        width = images[0].shape[2]
        r = postprocessing.detector_postprocess(predictions[0], height, width) # image

        bboxes = r.get("pred_boxes").tensor  # box
        classes = r.get("pred_classes")  # classes
        cls_probs = cls_probs[r_indices]  # clsporbs

        pooled_features = pooled_features[r_indices]

        if extract_mode == 1: # roi_feats

            assert (
                bboxes.size(0)
                == classes.size(0)
                == cls_probs.size(0)
                == pooled_features.size(0)
            )
            oriboxes = bboxes / batched_inputs[0]['im_scale']
            if not attr_scores is None:
                info = {
                    "objects": classes.cpu().numpy(),
                    "cls_prob": cls_probs.cpu().numpy(),
                    'attrs_id': attr_probs,
                    'attrs_scores': attr_scores,
                }
            else:
                # save info and features
                info = {
                    "objects": classes.cpu().numpy(),
                    "cls_prob": cls_probs.cpu().numpy(),
                }

            np.savez_compressed(
                os.path.join(dump_folder), 
                x=pooled_features.cpu().numpy(), 
                bbox=oriboxes.cpu().numpy(), 
                num_bbox=oriboxes.size(0), 
                image_h=image_h,
                image_w=image_w,
                image_h_inner=r.image_size[0], 
                image_w_inner=r.image_size[1],
                info=info
            )
        elif extract_mode == 2:  # bbox only
            oriboxes = bboxes / batched_inputs[0]['im_scale']
            np.savez_compressed(
                os.path.join(dump_folder), 
                bbox=oriboxes.cpu().numpy(), 
                num_bbox=oriboxes.size(0)
            )
        else:
            raise Exception("extract mode not supported:{}".format(extract_mode))
        
        if attribute_on:
            return [bboxes], [cls_probs], [pooled_features], [attr_probs]
        else:
            return [bboxes], [cls_probs], [pooled_features]
    else:  # extract mode == 3
        # postprocess
        height = images[0].shape[1]
        width = images[0].shape[2]

        if attribute_on:
            attr_scores = model.roi_heads.forward_attribute_score(pooled_features, cls_lables)
            attr_probs = F.softmax(attr_scores, dim=-1)

        bboxes = batched_inputs[0]['proposals'].proposal_boxes.tensor
        oriboxes = bboxes / batched_inputs[0]['im_scale']

        if not attr_scores is None:
            info = {
            "objects": cls_lables.cpu().numpy(),
            "cls_prob": cls_probs.cpu().numpy(),
            'attrs_id': attr_probs,
            'attrs_conf': attr_scores,
            }
        else:
            info = {
            "objects": cls_lables.cpu().numpy(),
            "cls_prob": cls_probs.cpu().numpy(),
            }

        np.savez_compressed(
            os.path.join(dump_folder), 
            x=pooled_features.cpu().numpy(), 
            bbox=oriboxes.cpu().numpy(), 
            num_bbox=oriboxes.size(0), 
            image_h=image_h,
            image_w=image_w,
            image_h_inner=height,
            image_w_inner=width,
            objects = cls_lables.cpu().numpy(),
            cls_prob = cls_probs.cpu().numpy(),
            attrs_id = attr_probs.cpu().numpy(),
            attrs_conf = attr_scores.cpu().numpy(),
        )
        
        if attribute_on:
            return [bboxes], [cls_probs], [pooled_features], [attr_probs]
        else:
            return [bboxes], [cls_probs], [pooled_features]

def extract_feat(split_idx, img_list, cfg, args):
    num_images = len(img_list)
    print('Number of images on split{}: {}.'.format(split_idx, num_images))

    model = DefaultTrainer.build_model(cfg)
    # print(model)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    model.eval()

    for im_file in (img_list):
        image_id = im_file.split('.')[0] # xxx.jpg
        dump_folder = os.path.join(args.output_dir, str(image_id) + ".npz")
        if os.path.exists(os.path.join(args.output_dir, im_file.split('.')[0]+'.npz')):
            continue

        im = cv2.imread(os.path.join(args.image_dir, im_file))
        illegal = False
        if im is None:
            illegal = True
        elif im.shape[-1] != 3:
            illegal = True
        elif max(im.shape[:2]) / min(im.shape[:2]) > 10 or max(im.shape[:2]) < 25:
            illegal = True
        if illegal:
            continue
        # dataset_dict = get_image_blob(im, cfg.MODEL.PIXEL_MEAN)
        pixel_mean = cfg.MODEL.PIXEL_MEAN if args.mode == "caffe" else 0.0
        image_h = np.size(im, 0)
        image_w = np.size(im, 1)
        dataset_dict = get_image_blob(im, pixel_mean)
        # extract roi features
        if cfg.MODEL.BUA.EXTRACTOR.MODE == 1:
            attr_scores = None
            with torch.set_grad_enabled(False):
                if cfg.MODEL.BUA.ATTRIBUTE_ON:
                    boxes, scores, features_pooled, attr_scores = model_inference(model,[dataset_dict], 1, dump_folder, image_h, image_w,True)
                else:
                    boxes, scores, features_pooled = model_inference(model, [dataset_dict], 1, dump_folder, image_h, image_w)
        elif cfg.MODEL.BUA.EXTRACTOR.MODE == 2:
            with torch.set_grad_enabled(False):
                boxes, scores, _ = model_inference(model,[dataset_dict], 2, dump_folder, image_h, image_w)
        elif cfg.MODEL.BUA.EXTRACTOR.MODE == 3: # extract roi features by bbox
            npy = False
            if os.path.exists(os.path.join(args.bbox_dir, im_file.split('.')[0]+'.npy')):
                npy = True
            elif not os.path.exists(os.path.join(args.bbox_dir, im_file.split('.')[0]+'.npz')):
                # actor.update.remote(1)
                continue
            if npy:
                try:
                    bbox = torch.from_numpy(np.load(os.path.join(args.bbox_dir, im_file.split('.')[0]+'.npy'), allow_pickle=True).tolist()['bbox']) * dataset_dict['im_scale']
                except Exception as e:
                    print(e)
                    continue
            else:
                bbox = torch.from_numpy(np.load(os.path.join(args.bbox_dir, im_file.split('.')[0]+'.npz'))['bbox']) * dataset_dict['im_scale']
            proposals = Instances(dataset_dict['image'].shape[-2:])
            proposals.proposal_boxes = BUABoxes(bbox)
            dataset_dict['proposals'] = proposals

            attr_scores = None
            with torch.set_grad_enabled(False):
                if cfg.MODEL.BUA.ATTRIBUTE_ON:
                    boxes, scores, features_pooled, attr_scores = model_inference(model,[dataset_dict], 3, dump_folder, image_h, image_w, True)
                else:
                    boxes, scores, features_pooled = model_inference(model,[dataset_dict], 3, dump_folder, image_h, image_w)
        else:
            raise Exception("extract mode not supported.")

def load_backbone(model, local_resume_path):
    checkpoint_state = torch.load(local_resume_path, map_location=lambda storage, loc: storage)
    model_state_dict = checkpoint_state['model_state_dict']

    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        if k.startswith("module."):
            name = k[7:]  # remove `module.`
        else:
            name = k
        #只load backbone
        if name.startswith("backbone."):
            name = name[9:]

        if "queue" in name:
            name = name + "_mis_match"
        new_state_dict[name] = v
    # print(new_state_dict.keys())
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    print("missing_keys:\n{}".format(missing_keys))
    print("unexpected_keys:\n{}".format(unexpected_keys))

if __name__ == "__main__":
    args = get_args()
    cfg = setup(args)

    extract_mode = 3
    if extract_mode !=3 :
        raise ValueError

    #load
    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
    model.eval()

    _, idx2path = get_ids()

    #personalized visual features for step3
    user2idx = {}
    with open('Public_Dataset/Data/idx/user2mmtrainidx.json', 'r') as f:
        user2mmtrainidx = json.load(f)
    f.close()

    user2idx = user2mmtrainidx
    with open('Public_Dataset/Data/idx/after1_user2testidx.json', 'r') as f:
        user2testidx = json.load(f)
    f.close()
    for user in user2testidx.keys():
        user2idx[user].extend(user2testidx[user])

    users_list = [str(i) for i in range(20)]
    for user in users_list:
        print(user, "*"*20)
        ids = user2idx[user]
        # load
        pth_save_root = '/Public_Dataset/Device_unimodal/checkpoint/users/lr0.0001_step50000_gamma0.99_adam/{}'.format(user)
        for fn in os.listdir(pth_save_root):
            if fn.startswith('9'):
                pth_name = fn
                break

        pth_path = os.path.join(pth_save_root,pth_name)
        load_backbone(model.backbone, pth_path)
        model.eval()

        for cnt, idx in tqdm(enumerate(ids),total=len(ids)):
            if cnt%1000 == 0:
                print(cnt)

            dump_folder =  os.path.join('/Public_Dataset/Data/two_stage/fine_npz', str(idx) + ".npz")

            npz_path = '/Public_Dataset/Data/npz/' + str(idx) + ".npz"
            npz_file = np.load(npz_path, allow_pickle=True)

            num_bbox = npz_file['num_bbox']
            bboxs = npz_file['bbox']
            image_h_inner = npz_file['image_h_inner']
            image_w_inner = npz_file['image_w_inner']

            img = cv2.imread(idx2path[idx])

            image_h = np.size(img, 0)
            image_w = np.size(img, 1)

            dataset_dict = get_image_blob(img)

            if extract_mode == 1:
                with torch.set_grad_enabled(False):
                    images = model.preprocess_image([dataset_dict])
                    features = model.backbone(images.tensor)
                    print("orig", features['res4'][0,0,0,:])
                    proposals, _ = model.proposal_generator(images, features, None)
                    _, pooled_features, _ = model.roi_heads.get_roi_features(features, proposals)  # fc7 feats
                    predictions = model.roi_heads.box_predictor(pooled_features)
                    cls_lables = torch.argmax(predictions[0], dim=1)
                    
                    cls_probs = F.softmax(predictions[0], dim=-1)
                    cls_probs = cls_probs[:, :-1]  # background is last
                    predictions, r_indices = model.roi_heads.box_predictor.inference(predictions, proposals)
                    
                    height = images[0].shape[1]
                    width = images[0].shape[2]
                    r = postprocessing.detector_postprocess(predictions[0], height, width) # image

                    bboxes = r.get("pred_boxes").tensor  # box
                    classes = r.get("pred_classes")  # classes
                    cls_probs = cls_probs[r_indices]  # clsporbs

                    pooled_features = pooled_features[r_indices]
                    print(pooled_features[0,:20])

                    assert (
                        bboxes.size(0)
                        == classes.size(0)
                        == cls_probs.size(0)
                        == pooled_features.size(0)
                    )
                    oriboxes = bboxes / dataset_dict['im_scale']
                    np.savez_compressed(
                        os.path.join(dump_folder), 
                        x=pooled_features.cpu().numpy(), 
                        bbox=oriboxes.cpu().numpy(), 
                        num_bbox=oriboxes.size(0), 
                        image_h=image_h,
                        image_w=image_w,
                        image_h_inner=r.image_size[0], 
                        image_w_inner=r.image_size[1],
                        objects= classes.cpu().numpy(),
                        cls_prob= cls_probs.cpu().numpy(),
                    )

            elif extract_mode == 3:
                with torch.set_grad_enabled(False):
                    images = model.preprocess_image([dataset_dict])
                    features = model.backbone(images.tensor)

                    proposals = Instances(dataset_dict['image'].shape[-2:])
                    bbox = bboxs * dataset_dict['im_scale']
                    proposals.proposal_boxes = BUABoxes(bbox)
                    dataset_dict['proposals'] = proposals

                    batched_inputs = [dataset_dict]
                    proposals = [x["proposals"].to(model.device) for x in batched_inputs]
                    _, pooled_features, _ = model.roi_heads.get_roi_features(features, proposals)  

                    predictions = model.roi_heads.box_predictor(pooled_features)
                    cls_lables = torch.argmax(predictions[0], dim=1)
                    
                    cls_probs = F.softmax(predictions[0], dim=-1)
                    cls_probs = cls_probs[:, :-1]  # background is last

                    bboxes = batched_inputs[0]['proposals'].proposal_boxes.tensor
                    oriboxes = bboxes / batched_inputs[0]['im_scale']

                    np.savez_compressed(
                        os.path.join(dump_folder), 
                        x=pooled_features.cpu().numpy(), 
                        bbox=oriboxes.cpu().numpy(), 
                        num_bbox=oriboxes.size(0), 
                        image_h=image_h,
                        image_w=image_w,
                        image_h_inner= image_h_inner, 
                        image_w_inner= image_w_inner,
                        objects= cls_lables.cpu().numpy(),
                        cls_prob= cls_probs.cpu().numpy(),
                    )

            else:
                raise ValueError

