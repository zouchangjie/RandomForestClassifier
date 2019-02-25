"""
Know more, visit my Python tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
tensorflow: 1.1.0
matplotlib
numpy
"""
import tensorflow as tf
import matplotlib.pyplot as plt
# coding: utf-8

# In[71]:

import time
import datetime
import numpy as np
import math
from sklearn.linear_model import LogisticRegression

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
    d3 = (int(day[1]))
    d4 = (int(day[2]))
    if d3==12 & d4==12 :
        continue
    elif d3==12 & d4==11:
        continue
    else:
        d2 = datetime.datetime(int(day[0]), int(day[1]), int(day[2]))
        diff_days = (d2 - d1).days
    # d2 = （int(day[1])，int(day[2])）
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

############# [Test Code]
# print ui_dict

# In[102]:

############# [Test Code]
# print len(ui_dict[0]), len(ui_dict[1]), len(ui_dict[2]), len(ui_dict[3])

# In[103]:

############# [Test Code]

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
y = np.reshape(y, [1531, 1])
y = [tf.one_hot(y, depth=2)]
y = tf.reshape(y, [1531, 2])
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

tf.set_random_seed(1)
np.random.seed(1)

# fake data

# x1 = np.linspace(-1, 1, 100)[:, np.newaxis]          # shape (100, 1)
# noise = np.random.normal(0, 0.1, size=x.shape)
# y1 = np.power(x, 2) + noise                           # shape (100, 1) + some noise

# plot data
# plt.scatter(x, y)
# plt.show()

tf_x = tf.placeholder(tf.float32, [None, 4])     # input x
tf_y = tf.placeholder(tf.int64, [None, 2])     # input y

# neural network layers
l1 = tf.layers.dense(tf_x, 10, tf.nn.relu)          # hidden layer
# l1 = tf.layers.dense(l1, 10)
# l1 = tf.layers.dense(l1, 10)
output = tf.layers.dense(l1, 2)                     # output layer
# output = tf.argmax(output, 1) # tf.nn.softmax(output)
# output = tf.reshape(output, [-1, 1])

loss = tf.losses.mean_squared_error(tf_y, output)   # compute cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train_op = optimizer.minimize(loss)

sess = tf.Session()                                 # control training and others
sess.run(tf.global_variables_initializer())         # initialize var in graph

# plt.ion()   # something about plotting

for step in range(1000):
    # train and net output
    y1 = sess.run(y)
    _, l, pred = sess.run([train_op, loss, output], {tf_x: x, tf_y: y1})
    # if step % 5 == 0:
    #     # plot and show learning process
    #     plt.cla()
    #     plt.scatter(x, y)
    #     plt.plot(x, pred, 'r-', lw=5)
    #     plt.text(0.5, 0, 'Loss=%.4f' % l, fontdict={'size': 20, 'color': 'red'})
    #     plt.pause(0.1)

# a = np.array([10] * 100)
# x = np.reshape(a, [100, 1])
# pred = sess.run([output], {tf_x: x})
# print(pred)
# print(px)
# fid = open('D:/result.txt', 'w', encoding='utf-8')
pred = sess.run([output], {tf_x: px})
predict = []
for i in range(len(pred[0])):
    if pred[0][i][0] > pred[0][i][1]:
        predict.append(0)
        # fid.write(str(0) + '\n')
    else:
        predict.append(1)
#         fid.write(str(1) + '\n')
# fid.close()

# npy = []
# pred1 = pred[0]
# for item in range(len(pred1)):
#     npy.append(pred1[item][0])
# x1 = npy

# x1 = np.reshape(pred, [1, len(pred)])
# print(p for p in pred[0])

lx = zip(train_data30,predict)
lx = sorted(lx, key=lambda x: x[1], reverse=True)

# In[140]:

############# [Test Code]


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
# fi = open('D:/vlaue.txt', 'w', encoding='utf-8')
# for i in range(len(lx)):
#     for j in range(len(lx[0])):
#         fi.write(str(lx[i][j]) + '\t')
#     fi.write('\n')
# fi.close()

size_predictionset = 0
for i in range(len(lx)):
    item = lx[i]
    if (item[1] == 1): size_predictionset += 1

size_referenceset = 0
for i in range(len(lx)):
    item = lx[i]
    if (item[0][2] == 4): size_referenceset += 1

size_predictionset_referenceset = 0
for i in range(len(lx)):
    item = lx[i]
    if (item[0][2] == 4) and (item[1] == 1): size_predictionset_referenceset += 1
    # if (item[1] == 0): break

P = 1.0 * size_predictionset_referenceset / size_predictionset * 100
print ('precision: %.2f%%' % (P))
R = 1.0 * size_predictionset_referenceset / size_referenceset * 100
print ('recal: %.2f%%' % (R))
F1 = 2.0 * P * R / (P + R)


print ('F1: %.2f%%' % (F1))
#tianchi_mobile_recommendation_predict

# pred3 = softmax(pred)
# print(pred3)
# plt.ioff()
# plt.show()