"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

convert image npz to LMDB
"""
import argparse
import glob
import io
import json
import multiprocessing as mp
import os
from os.path import basename, exists

from cytoolz import curry
import numpy as np
from tqdm import tqdm
import lmdb

import msgpack
import msgpack_numpy
import torch
import time
msgpack_numpy.patch()


def _compute_nbb(img_dump, conf_th, max_bb, min_bb, num_bb):
    num_bb = max(min_bb, (img_dump['conf'] > conf_th).sum())
    num_bb = min(max_bb, num_bb)
    return int(num_bb)


@curry
def load_npz(conf_th, max_bb, min_bb, num_bb, fname, keep_all=False):
    try:
        img_dump = np.load(fname, allow_pickle=True)
        if keep_all:
            nbb = None
        else:
            nbb = _compute_nbb(img_dump, conf_th, max_bb, min_bb, num_bb)
        dump = {}
        for key, arr in img_dump.items():
            if arr.dtype == np.float32:
                arr = arr.astype(np.float16)
            if arr.ndim == 2:
                dump[key] = arr[:nbb, :]
            elif arr.ndim == 1:
                dump[key] = arr[:nbb]
            else:
                raise ValueError('wrong ndim')
    except Exception as e:
        # corrupted file
        print(f'corrupted file {fname}', e)
        dump = {}
        nbb = 0

    name = basename(fname)
    return name, dump, nbb


def dumps_npz(dump, compress=False):
    with io.BytesIO() as writer:
        if compress:
            np.savez_compressed(writer, **dump, allow_pickle=True)
        else:
            np.savez(writer, **dump, allow_pickle=True)
        return writer.getvalue()


def dumps_msgpack(dump):
    return msgpack.dumps(dump, use_bin_type=True)

def NMS(dets, thresh):
    bxmin,bymin,bxmax,bymax,scores = dets[:,0],dets[:,1],dets[:,2],dets[:,3],dets[:,4]
    areas = (bxmax-bxmin+1) * (bymax-bymin+1)  #shape(n,) 
    order = scores.argsort()[::-1]  #
    keep = []  
    while order.size>0:
        i = order[0]  
        keep.append(i)
        #IOU
        xx = np.maximum(bxmin[i], bxmin[order[1:]])
        yy = np.maximum(bymin[i], bymin[order[1:]])
        XX = np.minimum(bxmax[i], bxmax[order[1:]])
        YY = np.minimum(bymax[i], bymax[order[1:]])
        w = np.maximum(0.0, XX-xx)
        h = np.maximum(0.0, YY-yy)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        idx = np.where(iou <= thresh)[0]  
        order = order[idx+1] 
    return keep

def normalize_bbox(bboxs, image_h, image_w):
    bbox_num = bboxs.shape[0]
    box_width = bboxs[:, 2] - bboxs[:, 0]
    box_height = bboxs[:, 3] - bboxs[:, 1]
    scaled_width = box_width / image_w
    scaled_height = box_height / image_h

    norm_bboxs = np.zeros((bbox_num, 6))
    norm_bboxs[:,:4] = bboxs
    norm_bboxs[:,0]/=image_w
    norm_bboxs[:,2]/=image_w
    norm_bboxs[:,1]/=image_h

    norm_bboxs[:,4] = norm_bboxs[:,2]-norm_bboxs[:,0] #w
    norm_bboxs[:,5] = norm_bboxs[:,3]-norm_bboxs[:,1] #h

    return norm_bboxs

#norm_bbox:xywh
def norm_and_sort(bboxs, image_h, image_w, cls_prob, nms_thres):
    num_bbox = bboxs.shape[0]

    max_conf = np.zeros(num_bbox)
    scores = cls_prob
    cls_boxes = bboxs
    for cls_ind in range(cls_prob.shape[1]): 
        cls_scores = scores[:, cls_ind]
        dets = np.hstack(
            (cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = np.array(NMS(dets, nms_thres))
        max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep],
                                    cls_scores[keep], max_conf[keep])

    confs_order = np.argsort(max_conf)[::-1] 
    max_confs_ordered = max_conf[confs_order]

    bboxs_ordered = bboxs[confs_order]
    norm_bboxs_ordered = normalize_bbox(bboxs_ordered, image_h=image_h, image_w=image_w) #[x,6]
    return confs_order, norm_bboxs_ordered, max_confs_ordered

@curry
def load_and_nms(opts, fn):
    idx= fn.split('.')[0]
    npz_path = os.path.join(opts.img_dir, fn)
    npz_file = np.load(npz_path, allow_pickle=True)

    num_bbox = npz_file['num_bbox']
    features = npz_file['x']
    bboxs = npz_file['bbox']
    image_h = npz_file['image_h']
    image_w = npz_file['image_w']
    cls_prob = npz_file['cls_prob']

    dump = {}
    features_in_order, norm_bb, max_confs_ordered = norm_and_sort(features, bboxs, image_h, image_w, cls_prob, opts.nms_thres)
    dump['features'] = features_in_order
    dump['norm_bb'] = norm_bb
    dump['conf'] = max_confs_ordered
    return idx, dump, num_bbox

@curry
def load_and_nms_base_and_fine(opts, fn):
    idx= fn.split('.')[0]
    base_npz_path = os.path.join(opts.base_img_dir, fn)
    fine_npz_path = os.path.join(opts.fine_img_dir, fn)
    base_npz_file = np.load(base_npz_path, allow_pickle=True)

    is_fine = os.path.exists(fine_npz_path)

    num_bbox = base_npz_file['num_bbox']
    try:
        base_features = base_npz_file['x']
    except:
        print(base_npz_path)
        quit()
    bboxs = base_npz_file['bbox']
    image_h = base_npz_file['image_h']
    image_w = base_npz_file['image_w']
    cls_prob = base_npz_file['cls_prob']

    dump = {}
    confs_order, norm_bb, max_confs_ordered = norm_and_sort( bboxs, image_h, image_w, cls_prob, opts.nms_thres)
    base_features_ordered = base_features[confs_order]

    dump['base_features'] = base_features_ordered
    dump['norm_bb'] = norm_bb
    dump['conf'] = max_confs_ordered

    if is_fine:
        fine_npz_file = np.load(fine_npz_path, allow_pickle=True)
        try:
            fine_features = fine_npz_file['x']
        except:
            print(fine_npz_path)
            quit()
        fine_features_ordered = fine_features[confs_order]
        dump['fine_features'] = fine_features_ordered

    return idx, dump, num_bbox

def main_mp_base_and_fine(opts):
    if not os.path.exists(opts.output):
        os.mkdir(opts.output)

    img_db_path = os.path.join(opts.output, "nms_thres_{}".format(str(opts.nms_thres)))
    if not os.path.exists(img_db_path):
        os.mkdir(img_db_path)

    db_name = 'all'

    env = lmdb.open(f'{img_db_path}/{db_name}', map_size=1024**4)
    txn = env.begin(write=True)
    
    idx2nbb = {}
    min_nbb = 4
    max_nbb = 100

    pool = mp.Pool(processes=opts.nproc)
    base_files = sorted(os.listdir(opts.base_img_dir))
    job = load_and_nms_base_and_fine(opts)

    idx2nbb_path = os.path.join(img_db_path, 'idx2nbb_all.json')
    if os.path.exists(idx2nbb_path):
        with open(idx2nbb_path, 'r') as f:
            idx2nbb = json.load(f)
        f.close()
    
    with open('/Public_Dataset/Data/idx/user2mmtrainidx.json', 'r') as f:
        user2mmtrainidx = json.load(f)
    f.close()
    mmtrainidx = []
    for user in user2mmtrainidx.keys():
        mmtrainidx.extend(user2mmtrainidx[user])


    with tqdm(total=len(base_files)) as pbar:
        for i, (idx, dump, nbb) in enumerate(
                pool.imap_unordered(job, base_files, chunksize=1)):
            if not idx:
                print("wrong!")
                continue  # corrupted feature

            dump = dumps_msgpack(dump)
            txn.put(key=idx.encode('utf-8'), value=dump)
            if i % 800 == 0:
                txn.commit()
                txn = env.begin(write=True)
                with open(idx2nbb_path, 'w') as f:
                    json.dump(idx2nbb,f)
                f.close()

            if i%2000 == 0:
                print("min nbb", min_nbb, "max nbb", max_nbb)

            if nbb < min_nbb:
                min_nbb = nbb
            if nbb > max_nbb:
                max_nbb = nbb
            idx2nbb[idx] = int(nbb)
            pbar.update(1)
        txn.commit()
        env.close()

    print("done! min nbb", min_nbb, "max nbb", max_nbb)
    with open(idx2nbb_path, 'w') as f:
        json.dump(idx2nbb,f)
    f.close()

def main_(opts):
    if opts.img_dir[-1] == '/':
        opts.img_dir = opts.img_dir[:-1]
    split = basename(opts.img_dir)
    if opts.keep_all:
        db_name = 'all'
    else:
        if opts.conf_th == -1:
            db_name = f'feat_numbb{opts.num_bb}'
        else:
            db_name = (f'feat_th{opts.conf_th}_max{opts.max_bb}'
                       f'_min{opts.min_bb}')
    if opts.compress:
        db_name += '_compressed'
    if not exists(f'{opts.output}/{split}'):
        os.makedirs(f'{opts.output}/{split}')
    env = lmdb.open(f'{opts.output}/{split}/{db_name}', map_size=1024**4)
    txn = env.begin(write=True)
    files = glob.glob(f'{opts.img_dir}/*.npz')
    load = load_npz(opts.conf_th, opts.max_bb, opts.min_bb, opts.num_bb,
                    keep_all=opts.keep_all)

    name2nbb = {}
    with mp.Pool(opts.nproc) as pool, tqdm(total=len(files)) as pbar:
        for i, (fname, features, nbb) in enumerate(
                pool.imap_unordered(load, files, chunksize=128)):
            if not features:
                continue  # corrupted feature
            if opts.compress:
                dump = dumps_npz(features, compress=True)
            else:
                dump = dumps_msgpack(features)
            txn.put(key=fname.encode('utf-8'), value=dump)
            if i % 1000 == 0:
                txn.commit()
                txn = env.begin(write=True)
            name2nbb[fname] = nbb
            pbar.update(1)
        txn.put(key=b'__keys__',
                value=json.dumps(list(name2nbb.keys())).encode('utf-8'))
        txn.commit()
        env.close()
    if opts.conf_th != -1 and not opts.keep_all:
        with open(f'{opts.output}/{split}/'
                  f'nbb_th{opts.conf_th}_'
                  f'max{opts.max_bb}_min{opts.min_bb}.json', 'w') as f:
            json.dump(name2nbb, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_img_dir", default='/Public_Dataset/Data/two_stage/base_npz', type=str,
                        help="The input images.")
    parser.add_argument("--fine_img_dir", default='/Public_Dataset/Data/two_stage/fine_npz', type=str,
                        help="The input images.")
    parser.add_argument("--output", default='/Public_Dataset/Data/img_db_two_stage', type=str,
                        help="output lmdb")
    parser.add_argument("--nms_thres", default=0.5, type=float,
                        help="output lmdb")
    parser.add_argument('--nproc', type=int, default=1,
                        help='number of cores used')
    parser.add_argument('--compress', action='store_true',
                        help='compress the tensors')
    parser.add_argument('--keep_all', action='store_true',
                        help='keep all features, overrides all following args')
    parser.add_argument('--conf_th', type=float, default=0.2,
                        help='threshold for dynamic bounding boxes '
                             '(-1 for fixed)')
    parser.add_argument('--max_bb', type=int, default=100,
                        help='max number of bounding boxes')
    parser.add_argument('--min_bb', type=int, default=10,
                        help='min number of bounding boxes')
    parser.add_argument('--num_bb', type=int, default=100,
                        help='number of bounding boxes (fixed)')
    args = parser.parse_args()

    main_mp_base_and_fine(args)
