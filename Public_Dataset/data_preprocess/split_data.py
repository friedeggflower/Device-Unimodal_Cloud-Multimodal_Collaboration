import json
import numpy as np 
import random
from collections import defaultdict


#  ============================  step 1  ============================ 
# data of step 1: cate2idx_1
with open('/Public_Dataset/Data/cate2idx.json', 'r') as f:
    cate2idx = json.load(f)
f.close()

cate2idx_1 = {} #step1
cate2idx_after1 = {} #step2-4
for cate in cate2idx.keys():
    cate_idxs = cate2idx[cate]
    _list = random.sample(cate2idx[cate], int(len(cate2idx[cate])/3))
    cate2idx_1[cate] = _list
    for idx in _list:
        cate_idxs.remove(idx)
    cate2idx_after1[cate] = cate_idxs

with open('/Public_Dataset/Data/idx/cate2idx_1.json', 'w') as f:
    json.dump(cate2idx_1, f)
f.close()

with open('/Public_Dataset/Data/idx/cate2idx_after1.json', 'w') as f:
    json.dump(cate2idx_after1, f)
f.close()

# ============================  after step 1: divide devices  ============================ 
# Each user obtains about 10,000 samples,
# Each user selects 5 classes and divides them equally (make sure each class is selected)
user_num = 20
cate_num = 5
user2cate = {}

cate_list = list(range(48))
cate_list = [str(cate) for cate in cate_list]

# Considering the small number of some categories, each category sets its own upper limit of choices
cate2pickthres = {}
for cate, idxs in cate2idx_after1.items():
    cate2pickthres[cate] = max(1, int(len(idxs)/400))

flag = False # make sure each class is selected
cate2usernum = defaultdict(lambda:0)
while not flag:
    for user in range(user_num):
        cate_flag = False
        while not cate_flag:
            _list = random.sample(cate_list, cate_num)
            cate_flag = True
            for _cate in _list:
                if not cate2usernum[_cate] < cate2pickthres[_cate]: 
                    cate_flag = False
        
        user2cate[user] = _list
        for cate in _list:
            cate2usernum[cate] += 1
    flag = True
    for cate, picked_num in cate2usernum.items():
        if picked_num == 0: #为0
            flag = False

cate2users = defaultdict(list)
for user, _list in user2cate.items():
    for cate in _list:
        cate2users[cate].append(user)

# divides the category equally between devices
user2idx = defaultdict(list)
idx2user = {}
for cate, users in cate2users.items():
    user_num = len(users)
    cate_idxs = cate2idx_after1[cate].copy()
    per_user_num = int(len(cate_idxs)/user_num)

    random.shuffle(users)
    for user_idx, user in enumerate(users):
        if user_idx != user_num-1:
            _idxs = random.sample(cate_idxs, per_user_num)
            user2idx[user].extend(_idxs)
            for idx in _idxs:
                cate_idxs.remove(idx)
        else: #the last user
            user2idx[user].extend(cate_idxs)

for user, idxs in user2idx.items():
    for idx in idxs:
        idx2user[idx] = user      

with open('/Public_Dataset/Data/idx/user2idx.json', 'w') as f:
    json.dump(user2idx, f)
f.close()

with open('/Public_Dataset/Data/idx/idx2user.json', 'w') as f:
    json.dump(idx2user, f)
f.close()

with open('/Public_Dataset/Data/idx/user2cate.json', 'w') as f:
    json.dump(user2cate, f)
f.close()


# ============================  step2-3 / step4  ============================ 
# divide testing and training for each device

user2testidx = {}
for key in user2idx.keys():
    _list = random.sample(user2idx[key], int(len(user2idx[key])/5)) #train:test = 4:1
    user2testidx[key] = _list
    for idx in _list:
        user2idx[key].remove(idx)

with open('/Public_Dataset/Data/idx/after1_user2trainidx.json', 'w') as f:
    json.dump(user2idx, f)
f.close()

with open('/Public_Dataset/Data/idx/after1_user2testidx.json', 'w') as f:
    json.dump(user2testidx, f)
f.close()


# ============================  step2 / step3  ============================ 
with open('/Public_Dataset/Data/idx/after1_user2trainidx.json', 'r') as f:
    user2trainidx = json.load(f)
f.close()

user2vistrainidx = {}
for user in user2trainidx.keys():
    _list = random.sample(user2trainidx[user], int(len(user2trainidx[user])/2))
    user2vistrainidx[user] = _list
    for idx in _list:
        user2trainidx[user].remove(idx)

with open('/Public_Dataset/Data/idx/user2vistrainidx.json', 'w') as f:
    json.dump(user2vistrainidx, f)
f.close()

with open('/Public_Dataset/Data/idx/user2mmtrainidx.json', 'w') as f:
    json.dump(user2trainidx, f)
f.close()


