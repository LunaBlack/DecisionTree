#!/usr/bin/env python
# -*- coding: UTF-8 -*-


# 代码来源（有修改）：http://jorbe.sinaapp.com/2014/07/13/decesion-tree-implement-with-python
# 该代码实现了决策树 C4.5算法分类
# 数据集为鸢尾花数据集

# 该代码未剪枝，从结果看属于过度拟合，测试的准确率略低



import math
import operator
from itertools import *
from numpy import *


# 取置信水平为0.95时的卡方表
CHI = {0.004,0.103,0.352,0.711,1.145,1.635,2.167,2.733,3.325,3.94,4.575,5.226,5.892,6.571,7.261,7.962}


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
    fr = open(filename)
    linenumber = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.strip().split()
        lineset = []
        for cel in listFromLine:
            lineset.append(cel)

        if(linenumber==0):
            labels=lineset
        else:
            dataSet.append(lineset)

        linenumber = linenumber+1
    return dataSet, labels


# 在给定特征和特征值的情况下，通过数组过滤的方式将数据集切分得到的两个子集返回
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:][0]
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:][0]
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
        subDataSet = splitDataSet(dataSet, axis , value)
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
        subDataSet0 = splitDataSet(mat0, axis , value)
        prob = len(subDataSet0)/float(len(dataSet))
        newGiniGain +=prob * calcGini(subDataSet0)
        subDataSet1 = splitDataSet(mat0, axis , value)
        prob = len(subDataSet1)/float(len(dataSet))
        newGiniGain +=prob * calcGini(subDataSet1)
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
        uniqueVals = set(featList)
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
    # the type is the same, so stop classify
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # traversal all the features and choose the most frequent feature
    if (len(dataSet[0]) == 1):
        return majorityCnt(classList)
    flag, bestFeat, bestSplit = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    if flag == 1:   #连续特征
        pass
# ------------------------------------------------ #
# 尚未完成 #
    #get the list which attain the whole properties
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


# 该函数用已建好的决策树进行分类
def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else: classLabel = secondDict[key]
    return classLabel



if __name__ == '__main__':
    myDat, myLabels = createDataSet()

    labels = []    # 该处有修改，为了避免createTree函数对labels的修改
    for each in myLabels:
        labels.append(each)

    myTree = createTree(myDat, labels)
    print myTree

    print classify(myTree, myLabels, [1,0])
    print classify(myTree, myLabels, [0,0])
    print classify(myTree, myLabels, [1,1])
