import h5py
import csv
from PIL import Image
import pickle
import json
import os

json_path = '/Public_Dataset/Data/meta.json'
meta = {}

h5_list = ['/Public_Dataset/Data/fashiongen_256_256_validation.h5', '/Public_Dataset/Data/fashiongen_256_256_train.h5']

# label for main category
cate2label = {}
cur_id = 0
cate2idx = {}

# label for sub category
subcate2label = {}
sub_cur_id = 0
subcate2idx = {}

for fn in h5_list:
    file_h5 = h5py.File(fn, 'r')
    for i in range(0,len(file_h5['index'])-1):
        if i%10000 == 0:
            print(i)   

        index = file_h5['index'][i][0]

        try:
            sub_cate = str(file_h5['input_subcategory'][i][0],'UTF-8')
            category = str(file_h5['input_category'][i][0],'UTF-8')
            description = str(file_h5['input_description'][i][0],'UTF-8')
        except Exception as e:
            try:
                sub_cate = str(file_h5['input_subcategory'][i][0],'latin-1')
                category = str(file_h5['input_category'][i][0],'latin-1')
                description = str(file_h5['input_description'][i][0],'latin-1')
            except:
                print("error decode")
                continue

        if "train" in fn:
            img_path = './train/image/' + str(index) + '.jpg'
        else:
            img_path = './valid/image/' + str(index) + '.jpg'

        if not os.path.exists(img_path):
            img = Image.fromarray(file_h5['input_image'][i])
            img.save(img_path)

        meta[str(index)] = {'img_path':img_path, 'description':description, 'cate':category}
        meta[str(index)]['subcate'] = sub_cate

        if category not in cate2label.keys():
            cate2label['category'] = cur_id
            cate2idx[cur_id] = [index]
            cur_id += 1
        else:
            cate2idx[cate2label[category]].append(index)

        if sub_cate not in subcate2label.keys():
            subcate2label[sub_cate] = sub_cur_id
            subcate2idx[str(sub_cur_id)] = [str(index)]
            sub_cur_id += 1
        else:
            subcate2idx[str(subcate2label[sub_cate])].append(str(index))


with open(json_path, 'w') as f:
    json.dump(meta, f)
f.close()

with open('/Public_Dataset/Data/fashion-gen_dataset/cate2label.json', 'w') as f:
    json.dump(cate2label, f)
f.close()

with open('/Public_Dataset/Data/cate2idx.json', 'w') as f:
    json.dump(cate2idx, f)
f.close()

with open('/Public_Dataset/Data/fashion-gen_dataset/sub_cate2label.json', 'w') as f:
    json.dump(subcate2label, f)
f.close()

with open('/Public_Dataset/Data/sub_label2idx.json', 'w') as f:
    json.dump(subcate2idx, f)
f.close()



