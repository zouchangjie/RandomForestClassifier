from sklearn import datasets
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from numpy import *

# iris = datasets.load_iris()
# X = iris.data
# Y = iris.target


def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = zeros((numberOfLines, 2))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:2]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


X, Y = file2matrix("data.txt")
shuxingname = ["x", "y"]
classname = ["1", "2"]

clf = RandomForestClassifier(n_estimators=5)
clf = clf.fit(X, Y)

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
    fo.write(shuxingname[num] + ", double,continuous,NA,NA,asMissing\n")
    print(shuxingname[num])

fo.write(str(numberClasses) + "\n")
for num in range(0, numberClasses):
    fo.write(classname[num] + "\n")
    print (classname[num])

fo.write(str(numberTrees) + "\n")
fo.close()
for num in range(0, numberTrees):
    fileName = "irsRF_" + str(num) + ".dot"
    with open(fileName, 'w') as f:
        f = tree.export_graphviz(Estimators[num], out_file=f)
