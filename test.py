# coding: utf-8

# In[71]:

import time
import datetime
import numpy as np
import math
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
# ############# reading data from files

# In[87]:

# with open('./tianchi_fresh_comp_train_user.csv') as f:
#    raw_data = f.read().splitlines()

controlsize = 100000

size = 0
raw_list = []
# for line in open("./tianchi_fresh_comp_train_user.csv"):
for line in open("./DataSet/tianchi_fresh_comp_train_user.csv"):
    if (size == 0):
        size += 1
        continue
    raw_list.append(line)
    size += 1
    if (size == controlsize): break
# raw_list =  raw_list.strip()
raw_list = map(lambda line: line.split(","), raw_list)

# In[88]:

############# [Test Code]
# print raw_list

# In[113]:

train_data = []
train_data28 = []
train_data29 = []
train_data30 = []
test = []
for line in raw_list:
    day = line[-1][:10].split('-')

    # 计算两个日期之间的天数
    d1 = datetime.datetime(2014, 11, 18)
    d2 = datetime.datetime(int(day[0]), int(day[1]), int(day[2]))
    diff_days = (d2 - d1).days

    uid = (int(line[0]), int(line[1]), int(line[2]), int(line[4]), diff_days)
    train_data.append(uid)

    if (diff_days <= 28):
        train_data28.append(uid)
    elif (diff_days == 29):
        train_data29.append(uid)
    elif (diff_days == 30):
        train_data30.append(uid)
    elif (diff_days > 30):
        test.append(uid)

train_data = list(set(train_data))
train_data28 = list(set(train_data28))
train_data29 = list(set(train_data29))
train_data30 = list(set(train_data30))

# In[90]:

############# [Test Code]
# print train_data

# In[114]:

############# [Test Code]
# print train_data28


# ############# data pre-processing

# In[99]:

def additem(uid, typeid, ui_dict, ui_buy):
    if uid in ui_dict[typeid]:
        ui_dict[typeid][uid] += 1
    else:
        ui_dict[typeid][uid] = 1
    if typeid == 3:
        ui_buy[uid] = 1
    return ui_dict, ui_buy


# In[100]:

# for feature
ui_dict = [{} for i in range(4)]
# for label
ui_buy = {}
for line in train_data:

    day = line[-1]
    if (day < 28): day = 28

    uid = (line[0], line[1], day)
    typeid = line[2] - 1
    ui_dict, ui_buy = additem(uid, typeid, ui_dict, ui_buy)

    for newday in range(day + 1, 31):
        uid = (line[0], line[1], newday)
        # print uid,
        ui_dict, ui_buy = additem(uid, typeid, ui_dict, ui_buy)

        # print ;

# In[101]:
# print '111'
############# [Test Code]
# print ui_dict

# In[102]:

############# [Test Code]
# print len(train_data), len(ui_dict)
# print len(ui_dict[0]), len(ui_dict[1]), len(ui_dict[2]), len(ui_dict[3])

# In[103]:

############# [Test Code]
# print ui_buy

# In[135]:

# get train X,Y
x = np.zeros((len(train_data29), 4))
y = np.zeros((len(train_data29),))

index = 0
for line in train_data29:
    uid = (line[0], line[1], line[-1] - 1)
    for i in range(4):
        x[index][i] = math.log1p(ui_dict[i][uid] if uid in ui_dict[i] else 0)
    uid = (line[0], line[1], line[-1])
    y[index] = 1 if uid in ui_buy else 0
    index += 1

# In[136]:

# get prediction px
px = np.zeros((len(train_data30), 4))

index = 0
for line in train_data30:
    uid = (line[0], line[1], line[-1] - 1)
    for i in range(4):
        px[index][i] = math.log1p(ui_dict[i][uid] if uid in ui_dict[i] else 0)
    index += 1

# In[120]:

############# [Test Code]
# print x
# print y
# print px

# ############# training

# In[137]:

model = LogisticRegression()
model.fit(x, y)
# model1 = tensokflow()
# ############# predicting

# In[138]:

py = model.predict_proba(px)
npy = []
for item in py:
    npy.append(item[1])
py = npy

# In[123]:

############# [Test Code]
# print py

# In[139]:

# combine and sort by predict score
lx = zip(train_data30, py)
lx = sorted(lx, key=lambda x: x[1], reverse=True)

# In[140]:

############# [Test Code]
# print lx

# In[130]:

wf = open('ans.csv', 'w')
wf.write('user_id,item_id\n')
for i in range(len(lx)):
    item = lx[i]
    if (item[1] < 0.5): break
    wf.write('%s,%s\n' % (item[0][0], item[0][1]))

wf.close()

# ############# evaluating

# In[141]:

size_predictionset = 0
for i in range(len(lx)):
    item = lx[i]
    if (item[1] >= 0.5): size_predictionset += 1

size_referenceset = 0
for i in range(len(lx)):
    item = lx[i]
    if (item[0][2] == 4): size_referenceset += 1

size_predictionset_referenceset = 0
for i in range(len(lx)):
    item = lx[i]
    if (item[0][2] == 4): size_predictionset_referenceset += 1
    if (item[1] < 0.5): break

P = 1.0 * size_predictionset_referenceset / size_predictionset * 100
R = 1.0 * size_predictionset_referenceset / size_referenceset * 100
F1 = 2.0 * P * R / (P + R)
print ('precision: %.2f%%' % (P))
print ('recal: %.2f%%' % (R))
print ('F1: %.2f%%' % (F1))

# In[ ]: