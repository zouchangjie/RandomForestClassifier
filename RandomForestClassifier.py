# from sklearn import datasets
# from sklearn import tree
# from sklearn.ensemble import RandomForestClassifier
# iris = datasets.load_iris()
# X = iris.data
# Y = iris.target
# clf = RandomForestClassifier(n_estimators=10)
# clf = clf.fit(X,Y)
#
# Estimators = clf.estimators_
# numberInputs = clf.feature_importances_
# numberTrees = len(clf.estimators_)
# fo = open("1.txt",'w')
# fo.write("RandomForestClassifier\n")
# fo.write("iRandomForestClassifier\n")
# fo.write("cRandomForestClassifier\n")
# fo.write("bRandomForestClassifier\n")
# fo.write(str(numberInputs)+"\n")
#
# for
#
# for num in range(0,numberTrees):
#     fileName
from csv import reader
from sklearn import datasets
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset
filename = 'ans.csv'
dataset = load_csv(filename)
x=[]
# x1=[]
y=[]
for item in dataset :
    x1=map(float, item[:-1])
    x.append(x1)
# x = map(eval, x)
# x1 = ['2.22','1.11']
# x2 = map(float, x1)

for item in dataset :
    x1 = map(int, item[2:3])
    x1=x1[0]
    y.append(x1)
# iris = datasets.load_iris()
iris = ["x", "y"]
classname = ["1", "2"]

# X = iris.data
# Y = iris.target
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(x, y)
# export model params
Estimators = clf.estimators_
Importances = clf.feature_importances_

numberClasses = clf.n_classes_
numberInputs = len(clf.feature_importances_)
numberTrees = len(clf.estimators_)

fo = open("RandomForestModel.txt", "w")
fo.write("RandomForestClassifier\n")
fo.write("IrisRandomForestModel\n")
fo.write("classification\n")
fo.write("binarySplit\n")

fo.write(str(numberInputs) + "\n")
for num in range(0, numberInputs):
    fo.write(iris[num] + ", double,continuous,NA,NA,asMissing\n")

fo.write(str(numberClasses) + "\n")
for num in range(0, numberClasses):
    fo.write(iris[num] + "\n")
fo.write(str(numberTrees) + "\n")
fo.close()
for num in range(0, numberTrees):
    fileName = "irsRF_" + str(num) + ".dot"
    with open(fileName, 'w') as f:
        f = tree.export_graphviz(Estimators[num].tree_, out_file=f)
X=np.array([[1.12,1.122]])
y = clf.predict(X=X)
# clf.predict_proba()
print(y)
# print (outputs)