#!/usr/bin/env python
# -*- coding: UTF-8 -*-


# 该代码实现了决策树 CART算法分类
# 算法思想参考1：http://blog.csdn.net/tianguokaka/article/details/9018933
# 算法思想参考2：http://blog.csdn.net/suipingsp/article/details/42264413
# 数据集为鸢尾花数据集（部分）

# 该代码已实现剪枝，但部分未完善



import string
import math
import operator
from itertools import *
from numpy import *
from treePlotter import createPlot
from pprint import pprint



# 该函数计算给定Gini：Gini越大，信息越不容易搞清楚
# ！该处是与ID3、C4.5算法不同的地方
def calcGini(dataSet):
    #calculate the Gini value
    num = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:      #create the dictionary for all of the data
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    gini = 1.0
    for key in labelCounts:
        prob = float(labelCounts[key])/num
        gini -= prob**2
    return gini


# 该函数为创建数据的函数。实际情况中，该函数需要改造成读取训练数据的函数
def createDataSet():
    dataSet = [[1,1,'yes'],
               [1,1, 'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet, labels


# 该函数从txt文本创建数据集
def createDataSetFromTXT(filename):
    dataSet = []
    labels = []
    fr = open(filename, 'r')
    linenumber = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.strip().split(',')
        lineset = []
        for each in listFromLine:
            if type(each) == str:
                try:
                    each = float(each)
                except:
                    pass
            lineset.append(each)
        if linenumber == 0:
            labels = lineset
        else:
            dataSet.append(lineset)
        linenumber = linenumber + 1
    return dataSet, labels


# 在给定特征和特征值的情况下，通过数组过滤的方式将数据集切分得到的两个子集返回
def binSplitDataSet(dataSet, feature, value):
    mat0 = []
    mat1 = []
    for each in dataSet:
        if each[feature] > value:
            mat0.append(each)
        elif each[feature] <= value:
            mat1.append(each)
    return mat0, mat1


# 处理离散分布、且取值数目>=3的特征，返回特征值的全部组合方式
# 利用itertools包创建多值离散特征二分序列组合
def featuresplit(features):
    count = len(features)
    featureind = range(count)
    featureind.pop(0)    #get value 1~(count-1)
    combiList = []
    for i in featureind:
        com = list(combinations(features, len(features[0:i])))
        combiList.extend(com)
    combiLen = len(combiList)
    featuresplitGroup = zip(combiList[0:combiLen/2], combiList[combiLen-1:combiLen/2-1:-1])
    return featuresplitGroup


# 该函数根据给定的特征（即属性）划分数据集
# dataSet：待划分的数据集
# axis：划分数据集的特征（即属性）--数据的第几列
# valueTuple：划分的特征值（即属性的值域）
# return dataset satisfy condition dataSet[i][axis] == valueTuple
# remove dataSet[i][axis] if len(valueTuple)==1
def splitDataSet(dataSet, axis, valueTuple):
    retDataSet = []
    length = len(valueTuple)
    if length == 1:
        for featVec in dataSet:
            if featVec[axis] == valueTuple[0]:
                reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
                reducedFeatVec.extend(featVec[axis+1:])
                retDataSet.append(reducedFeatVec)
    else:
        for featVec in dataSet:
            if featVec[axis] in valueTuple:
                retDataSet.append(featVec)
    return retDataSet


# 该函数选择最好的数据集划分方式，针对离散特征，且特征只有两种取值
# 返回特定特征的GiniGain
def chooseBestSplit0(dataSet, axis, values):
    newGiniGain = 0.0
    for value in values:
        subDataSet = splitDataSet(dataSet, axis , [value])
        prob = len(subDataSet)/float(len(dataSet))
        newGiniGain +=prob * calcGini(subDataSet)
    return newGiniGain


# 该函数选择最好的数据集划分方式，针对离散特征，且特征至少三种取值
# 返回特定特征的GiniGain
def chooseBestSplit1(dataSet, axis, values):
    featuresplitGroup = featuresplit(values)
    bestGiniGain = 2.0
    featureSplitList = []
    for each in featuresplitGroup:
        newGiniGain = 0.0
        for value in each:
            if len(value) == 1:
                subDataSet = splitDataSet(dataSet, axis , [value])
            else:
                subDataSet = splitDataSet(dataSet, axis , value)
            prob = len(subDataSet)/float(len(dataSet))
            newGiniGain +=prob * calcGini(subDataSet)
        if newGiniGain < bestGiniGain:
            bestGiniGain = newGiniGain
            featureSplitList = each
    return newGiniGain, featureSplitList


# 找到数据的最佳二元切分方式，针对连续特征
def chooseBestSplit2(dataSet, axis, values):
    bestValue = inf    #正无穷
    bestGiniGain = 2.0
    for value in values:
        newGiniGain = 0.0
        mat0, mat1 = binSplitDataSet(dataSet, axis, value)
        prob = len(mat0)/float(len(dataSet))
        newGiniGain +=prob * calcGini(mat0)
        prob = len(mat1)/float(len(dataSet))
        newGiniGain +=prob * calcGini(mat1)
        if newGiniGain < bestGiniGain:
            bestGiniGain = newGiniGain
            bestValue = value
    return bestGiniGain, bestValue


# 找到数据的最佳二元切分方式
# 根据特征是离散还是连续，采用不同的方法
# 通过计算GiniGain，选择GiniGain最小的特征（属性），以此特征作为划分数据集的方式。
# GiniGain越小，该特征越适于分类
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    bestGiniGain = 2.0
    bestFeature = -1
    bestFlag = 0
    splitValue = inf
    featureSplitList = []
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = list(set(featList))
        flag = 0
        if len(uniqueVals) > 6:   #取值多于6个，判断为连续特征
            newGiniGain, newSplitValue = chooseBestSplit2(dataSet, i, uniqueVals)
            flag = 1
        else:    #取值少于6个，判断为离散特征
            if len(uniqueVals) == 2:
                newGiniGain = chooseBestSplit0(dataSet, i, uniqueVals)
                newFeatureSplitList = [value for value in uniqueVals]
                flag = 2
            elif len(uniqueVals) >= 3:
                newGiniGain, newFeatureSplitList = chooseBestSplit1(dataSet, i, uniqueVals)
                flag = 3
        if(newGiniGain < bestGiniGain):
            bestGiniGain = newGiniGain
            bestFeature = i
            bestFlag = flag
            if flag == 1:
                splitValue = newSplitValue
            elif flag == 2 or flag == 3:
                featureSplitList = newFeatureSplitList
    if bestFlag == 1:
        return bestFlag, bestFeature, splitValue
    elif bestFlag == 2 or bestFlag == 3:
        return bestFlag, bestFeature, featureSplitList


# 该函数用于找出出现次数最多的分类名称
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    # 利用operator操作键值排序字典
    # iteritems迭代器返回键值对，operator.itemgetter选择排序的特征，reverse反向
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 该函数递归地创建树
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):   #属于同一类
        return classList[0]
    if (len(dataSet[0]) == 1):   #只有一个特征
        return majorityCnt(classList)
    # if len(dataSet) < 5:    #待划分的样本个数小于阈值5
    #     return majorityCnt(classList)
    flag, bestFeat, bestSplit = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    if flag == 1:   #连续特征
        mat0, mat1 = binSplitDataSet(dataSet, bestFeat, bestSplit)
        myTree[bestFeatLabel]['>%f' % bestSplit] = createTree(mat0, labels)
        myTree[bestFeatLabel]['<=%f' % bestSplit] = createTree(mat1, labels)
    elif flag == 2:   #只有两个取值的离散特征
        del(labels[bestFeat])
        subLabels = labels[:]
        myTree[bestFeatLabel][bestSplit[0]] = createTree(splitDataSet(dataSet, bestFeat, [bestSplit[0]]), subLabels)
        myTree[bestFeatLabel][bestSplit[1]] = createTree(splitDataSet(dataSet, bestFeat, [bestSplit[1]]), subLabels)
    elif flag == 3:   #至少三个取值的离散特征
        for i in bestSplit:
            if len(i) == 1:
                subLabels = labels[:]
                subLabels.pop(bestFeat)
                myTree[bestFeatLabel][i] = createTree(splitDataSet(dataSet, bestFeat, i), subLabels)
            elif len(i) > 1:
                subLabels = labels[:]
                myTree[bestFeatLabel][i] = createTree(splitDataSet(dataSet, bestFeat, i), subLabels)
    return myTree


# 该函数用已建好的决策树进行分类
def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if type(key) == str and key.startswith('>'):
            value = float(key[1:])
            if testVec[featIndex] > value:
                if type(secondDict[key]).__name__ == 'dict':
                    classLabel = classify(secondDict[key], featLabels, testVec)
                else:
                    classLabel = secondDict[key]
        elif type(key) == str and key.startswith('<='):
            value = float(key[2:])
            if testVec[featIndex] <= value:
                if type(secondDict[key]).__name__ == 'dict':
                    classLabel = classify(secondDict[key], featLabels, testVec)
                else:
                    classLabel = secondDict[key]
        elif type(key) == int or type(key) == float:
            if testVec[featIndex] == key:
                if type(secondDict[key]).__name__ == 'dict':
                    classLabel = classify(secondDict[key], featLabels, testVec)
                else:
                    classLabel = secondDict[key]
        elif type(key) == tuple:
            if testVec[featIndex] in key:
                if type(secondDict[key]).__name__ == 'dict':
                    classLabel = classify(secondDict[key], featLabels, testVec)
                else:
                    classLabel = secondDict[key]
    return classLabel


class Node():
    def __init__(self, dictitem, featLabels):
        if isinstance(dictitem, dict):
            self.isLeaf = False
            self.featLabel, b = dictitem.items()[0]
            self.featIndex = featLabels.index(self.featLabel)
            self.condition = ''
            pre = True
            for condition, d in b.items():
                self.condition += str(condition)
                if isinstance(condition, tuple):
                    if pre:
                        leftcondition = condition
                        self.leftsatisfy = lambda x: x[self.featIndex] in leftcondition
                        self.leftNode = Node(d, featLabels)
                    else:
                        rightcondition = condition
                        self.rightsatisfy = lambda x: x[self.featIndex] in rightcondition
                        self.rightNode = Node(d, featLabels)

                elif isinstance(condition, str):
                    if condition.startswith('<='):
                        thre = float(condition[2:])
                        if pre:
                            self.leftsatisfy = lambda x: x[self.featIndex] <= thre
                            self.leftNode = Node(d, featLabels)
                        else:
                            self.rightsatisfy = lambda x: x[self.featIndex] <= thre
                            self.rightNode = Node(d, featLabels)
                    elif condition.startswith('>'):
                        thre = float(condition[1:])
                        if pre:
                            self.leftsatisfy = lambda x: x[self.featIndex] > thre
                            self.leftNode = Node(d, featLabels)
                        else:
                            self.rightsatisfy = lambda x: x[self.featIndex] > thre
                            self.rightNode = Node(d, featLabels)
                pre = False
            self.leftNode.setParent(self)
            self.rightNode.setParent(self)

        else:
            self.isLeaf = True
            self.classresult = dictitem

        self.nleft = 0
        self.nright = 0

        self.alpha = 0

    def setParent(self, parent):
        self.parent = parent

    def classify(self, dataset):
        leftSet = []
        rightSet = []

        for e in dataset:
            if self.leftsatisfy(e):
                leftSet.append(e)
                self.nleft += 1
            elif self.rightsatisfy(e):
                rightSet.append(e)
                self.nright += 1
            else:
                print(self.condition)
                import ipdb; ipdb.set_trace()
                print("!"*10, e)

        print(self.condition, self.nleft, self.nright)

        if self.leftNode.isLeaf == False:
            self.leftNode.classify(leftSet)
        else:
            print(self.leftNode.classresult, self.nleft)
            for e in leftSet:
                # assert e[-1] == self.leftNode.classresult
                if e[-1] != self.leftNode.classresult:
                    print(e, self.leftNode.classresult)

        if self.rightNode.isLeaf == False:
            self.rightNode.classify(rightSet)
        else:
            print(self.rightNode.classresult, self.nright)
            for e in rightSet:
                # assert e[-1] == self.rightNode.classresult
                if e[-1] != self.rightNode.classresult:
                    print(e, self.rightNode.classresult)

    def calcuAlpha(self, totalNumber):
        if self.isLeaf:
            return 0, 1
        else:
            lalpha2, lsize = self.leftNode.calcuAlpha(totalNumber)
            ralpha2, rsize = self.rightNode.calcuAlpha(totalNumber)
            self.leafsize = lsize + rsize

            self.alpha = (min(self.nleft, self.nright)/float(totalNumber) - lalpha2 - ralpha2) \
                    /(lsize + rsize -1)
            print(self.alpha)
            return (lalpha2 + ralpha2), (lsize + rsize)


def pruning(tree, number):
    root = tree

    while number:
        number -= 1

        s = []
        minalpha = 1
        minNode = root

        while not root.isLeaf or s:
            if root.isLeaf:
                root = s.pop()
                root = root.rightNode
            else:
                if root.alpha < minalpha or (root.alpha == minalpha and root.leafsize < minNode.leafsize):
                    minNode = root
                    minalpha = minNode.alpha
                s.append(root)
                root = root.leftNode

        print("will prune node", minNode.condition)
        minNode.isLeaf = True
        minNode.leftNode = None
        minNode.rightNode = None

        # tree.calcuAlpha()


# 主函数
def run(train_file, test_file):
    train_dataset, myLabels = createDataSetFromTXT(train_file)
    # import ipdb; ipdb.set_trace()

    labels = []
    for each in myLabels:
        labels.append(each)
    decisionTree = createTree(train_dataset, labels)
    print 'decisionTree :'
    pprint(decisionTree)

    # createPlot(decisionTree)

    # leaf, errGain, finalTree = pruning(train_dataset, decisionTree, myLabels)

    newTree = Node(decisionTree, myLabels)
    newTree.classify(train_dataset)
    newTree.calcuAlpha(len(train_dataset))

    pruning(newTree, 1)

    # newTree.classify(train_dataset)  # 剪枝后的树

    test_dataset, testLabels = createDataSetFromTXT(test_file)
    n = len(test_dataset)
    correct = 0
    for test_data in test_dataset:
        label = classify(decisionTree, myLabels, test_data[:-1])
        if label == test_data[-1]:
            correct += 1
    print "准确率: ".decode('utf8'), correct/float(n)



if __name__ == '__main__':
    run('irisTrain.txt', 'irisTest.txt')

    myDat, myLabels = createDataSet()

    labels = []
    for each in myLabels:
        labels.append(each)

    myTree = createTree(myDat, labels)
    print myTree

    print classify(myTree, myLabels, [1,0])
    print classify(myTree, myLabels, [0,0])
    print classify(myTree, myLabels, [1,1])

