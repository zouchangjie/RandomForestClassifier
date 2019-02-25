import numpy as np
import matplotlib.pyplot as plt
# x = np.linspace(-1, 1, 50)[:, np.newaxis]          # shape (100, 1)
# noise = np.random.normal(0, 0.2, size=x.shape)
# y = np.power(x, 2) + noise
#  shape (100, 1) + some noise
x=np.random.uniform(-1,1,100)
wf = open('ans.csv', 'w')
# wf.write('x\ty\n')
# for i in range(len(x)):
#     item = str(x[i])
#     wf.write(item)
#     wf.write('\n')

y = []
for i in range(len(x)):
    y.append(np.random.uniform(-np.sqrt(1-np.power(x[i], 2)), np.sqrt(1-np.power(x[i], 2))))
    item = str(x[i])
    wf.write(str(item)+","+str(y[i])+","+'0')
    wf.write('\n')
plt.scatter(x, y,c='r')

x=np.random.uniform(-1.1,1.1,50)
y = []
for i in range(len(x)):
    y.append(np.random.uniform(np.sqrt(1.3-np.power(x[i], 2)), np.sqrt(1.7-np.power(x[i], 2))))
    item = str(x[i])
    wf.write(str(item) + "," + str(y[i])+","+'1')
    wf.write('\n')

plt.scatter(x, y,c='b')
y=[]
for i in range(len(x)):
    y.append(np.random.uniform(-np.sqrt(1.7-np.power(x[i], 2)),-np.sqrt(1.3-np.power(x[i], 2))))
    item = str(x[i])
    wf.write(str(item) + "," + str(y[i])+","+'1')
    wf.write('\n')
wf.close()
plt.scatter(x, y,c='b')
plt.show()

