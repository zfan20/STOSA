import gzip
from collections import defaultdict
from datetime import datetime
import os
import copy
import json


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


countU = defaultdict(lambda: 0)
countP = defaultdict(lambda: 0)
line = 0

DATASET = 'Office_Products'
dataname = '/home/zfan/BDSC/projects/datasets/reviews_' + DATASET + '_5.json.gz'
if not os.path.isdir('./'+DATASET):
    os.mkdir('./'+DATASET)
train_file = './'+DATASET+'/train.txt'
valid_file = './'+DATASET+'/valid.txt'
test_file = './'+DATASET+'/test.txt'
imap_file = './'+DATASET+'/imap.json'
umap_file = './'+DATASET+'/umap.json'

data_file = './'+DATASET+'.txt'



for one_interaction in parse(dataname):
    rev = one_interaction['reviewerID']
    asin = one_interaction['asin']
    time = float(one_interaction['unixReviewTime'])
    countU[rev] += 1
    countP[asin] += 1

usermap = dict()
usernum = 1
itemmap = dict()
itemnum = 1
User = dict()


for one_interaction in parse(dataname):
    rev = one_interaction['reviewerID']
    asin = one_interaction['asin']
    time = float(one_interaction['unixReviewTime'])
    if countU[rev] < 5 or countP[asin] < 5:
        continue

    if rev in usermap:
        userid = usermap[rev]
    else:
        userid = usernum
        usermap[rev] = userid
        User[userid] = []
        usernum += 1
    if asin in itemmap:
        itemid = itemmap[asin]
    else:
        itemid = itemnum
        itemmap[asin] = itemid
        itemnum += 1
    User[userid].append([itemid, time])
# sort reviews in User according to time


with open(imap_file, 'w') as f:
    json.dump(itemmap, f)

with open(umap_file, 'w') as f:
    json.dump(usermap, f)

for userid in User.keys():
    User[userid].sort(key=lambda x: x[1])


user_train = {}
user_valid = {}
user_test = {}
for user in User:
    nfeedback = len(User[user])
    if nfeedback < 3:
        user_train[user] = User[user]
        user_valid[user] = []
        user_test[user] = []
    else:
        user_train[user] = User[user][:-2]
        user_valid[user] = []
        user_valid[user].append(User[user][-2])
        user_test[user] = []
        user_test[user].append(User[user][-1])



print(usernum, itemnum)

def writetofile(data, dfile):
    with open(dfile, 'w') as f:
        for u, ilist in sorted(data.items()):
            for i, t in ilist:
                f.write(str(u) + '\t'+ str(i) + '\t' + str(t) + "\n")

def writetofile_v2(data, dfile):
    with open(dfile, 'w') as f:
        for u, ilist in sorted(data.items()):
            f.write(str(u))
            for i, t in ilist:
                f.write(' '+ str(i))
            f.write("\n")

#writetofile(user_train, train_file)
#writetofile(user_valid, valid_file)
#writetofile(user_test, test_file)

writetofile_v2(User, data_file)



num_instances = sum([len(ilist) for _, ilist in User.items()])
print('total user: ', len(User))
print('total instances: ', num_instances)
print('avg length: ', num_instances / len(User))
print('total items: ', itemnum)
print('density: ', num_instances / (len(User) * itemnum))
print('valid #users: ', len(user_valid))
numvalid_instances = sum([len(ilist) for _, ilist in user_valid.items()])
print('valid instances: ', numvalid_instances)
numtest_instances = sum([len(ilist) for _, ilist in user_test.items()])
print('test #users: ', len(user_test))
print('test instances: ', numtest_instances)
