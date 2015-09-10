#!/usr/bin/env python
# -*- coding: UTF-8 -*-


# 代码来源：http://blog.csdn.net/alvine008/article/details/37760639（machine learning in action 第三章例子）
# 代码说明还可以参照：http://www.cnblogs.com/hantan2008/archive/2015/07/27/4674097.html
# 该代码实现了决策树算法分类


import math
import operator


# 该函数计算给定数据集的信息熵（香农熵）：信息熵越大，信息越不容易搞清楚
def calcShannonEnt(dataSet):
    #calculate the shannon value
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:      #create the dictionary for all of the data
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * math.log(prob,2) #get the log value
    return shannonEnt


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
    dataSet = []; labels = []
    fr = open(filename)
    linenumber=0
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


# 该函数根据给定的特征（即属性）划分数据集
# dataSet：待划分的数据集
# axis：划分数据集的特征（即属性）--数据的第几列
# value：需要返回的特征值（即属性的值域）
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:      #abstract the fature
            reducedFeatVec = featVec[:axis]    #获取从第0列到特征列的数据
            reducedFeatVec.extend(featVec[axis+1:])    #获取从特征列之后的数据
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 该函数选择最好的数据集划分方式
# 通过计算信息增益Gain，选择信息增益最大的特征（属性），以此特征作为划分数据集的方式。
# 信息增益越大，该特征越适于分类
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0])-1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i , value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy +=prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


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
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
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