"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

preprocess annotations into LMDB
"""
import argparse
import json
import pickle
import os,shutil
from os.path import exists

from cytoolz import curry
from tqdm import tqdm
from pytorch_pretrained_bert import BertTokenizer

from data.data import open_lmdb
import nltk
from nltk.stem import WordNetLemmatizer
import time

wnl = WordNetLemmatizer()

wrong_token = 0

@curry
def bert_tokenize(tokenizer, text):
    ids = []
    for word in text.strip().split():
        ws = tokenizer.tokenize(word)
        if not ws:
            # some special char
            wrong_token+=1
            if wrong_token%100==0:
                print("wrong_token num", wrong_token)

            continue
        ids.extend(tokenizer.convert_tokens_to_ids(ws))
    return ids

#mask 文本里的cate里的关键词
@curry
def tokenize_mask(tokenizer, text, mask_list):
    ids = []
    for word in text.strip().split():
        if word in mask_list:
            ids.extend([103])
            print(word, "masked")
        else:
            ws = tokenizer.tokenize(word)
            if not ws:
                # some special char
                continue
            ids.extend(tokenizer.convert_tokens_to_ids(ws))
    return ids

#mask 文本里的cate里的关键词
@curry
def tokenize_stem_mask(tokenizer, text, wnl, stem_list):
    ids = []
    flag = False
    for word in text.strip().split():
        if wnl.lemmatize(word.lower(),'n') in stem_list:
            ids.extend([103])
            flag = True
        else:
            ws = tokenizer.tokenize(word)
            if not ws:
                # some special char
                continue
            ids.extend(tokenizer.convert_tokens_to_ids(ws))
    return ids, flag

def plural(word):
    if word.endswith('y'):
        return word[:-1] + 'ies'
    elif word[-1] in 'sx' or word[-2:] in ['sh', 'ch']:
        return word + 'es'
    elif word.endswith('an'):
        return word[:-2] + 'en'
    else:
        return word + 's'


def main(opts):
    if not exists(opts.output):
        os.makedirs(opts.output)

    meta = vars(opts)
    meta['tokenizer'] = opts.toker
    toker = BertTokenizer.from_pretrained(opts.toker, do_lower_case='uncased' in opts.toker)

    tokenizer = tokenize_stem_mask(toker)
    meta['UNK'] = toker.convert_tokens_to_ids(['[UNK]'])[0]
    meta['CLS'] = toker.convert_tokens_to_ids(['[CLS]'])[0]
    meta['SEP'] = toker.convert_tokens_to_ids(['[SEP]'])[0]
    meta['MASK'] = toker.convert_tokens_to_ids(['[MASK]'])[0]
    meta['v_range'] = (toker.convert_tokens_to_ids('!')[0],
                       len(toker.vocab))
    
    if not os.path.exists('{opts.output}/meta.json'):
        with open(f'{opts.output}/meta.json', 'w') as f:
            json.dump(vars(opts), f, indent=4)

    wnl = WordNetLemmatizer()
    #======================================  tokenize subcate_label ======================================
    if not os.path.exists('{}/raw_mask_stem_subcate_target.json'.format(opts.output)):
        with open('Public_Dataset/Data/txt_db_mask_cate/raw_mask_stem_target.json','r')as f:
            subcate_stem_list = json.load(f)
        f.close()
        #add some missing plurals manually
        for subcate in ['crewneck','sweatpant','hoody','zipup','v-neck','henley','bodysuit','shawlneck','up','monkstrap']:
            if subcate not in subcate_stem_list:
                subcate_stem_list.append(subcate)

        with open('Public_Dataset/Data/sub_cate2label.json', 'r')as f:
            subcate2label = json.load(f)
        f.close()
        raw_subcate_list = list(subcate2label.keys())
        # subcate_stem_list= []
        for subcate in raw_subcate_list:
            _list = subcate.split(" ")
            for word in _list:
                if len(word) > 1:
                    _stem_cate = wnl.lemmatize(word.lower(), 'n')
                    if _stem_cate not in subcate_stem_list:
                        subcate_stem_list.append(_stem_cate)
        
        print(subcate_stem_list)
        with open('{}/raw_mask_stem_subcate_target.json'.format(opts.output), 'w') as f:
            json.dump(subcate_stem_list,f)
        f.close()
    else:
        with open('{}/raw_mask_stem_subcate_target.json'.format(opts.output), 'r') as f:
            subcate_stem_list = json.load(f)
        f.close()

    open_db = curry(open_lmdb, opts.output, readonly=False)

    anno_path = 'Public_Dataset/Data/meta.json'
    mask_cnt = 0
    with open_db() as db:
        with open(anno_path, 'r') as f:
            ann = json.load(f)
        f.close()
        id2len = {}

        for cnt,idx in enumerate(ann.keys()):
            input_ids, masked = tokenizer(ann[idx]['description'], wnl, subcate_stem_list)
            if masked:
                mask_cnt +=1


            db[idx] = input_ids
            id2len[idx] = len(input_ids)

    with open(f'{opts.output}/id2len.json', 'w') as f:
        json.dump(id2len,f)
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--output', default='Public_Dataset/Data/txt_db_mask_subcate',
                        help='output dir of DB')

    parser.add_argument('--toker', default='bert-base-cased',
                        help='which BERT tokenizer to used')
    args = parser.parse_args()
    main(args)
