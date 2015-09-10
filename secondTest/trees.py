#!/usr/bin/env python
# -*- coding: UTF-8 -*-


# 代码来源：http://www.cnblogs.com/hantan2008/archive/2015/07/27/4674097.html
# 该代码实现了决策树算法分类（ID3算法）
# 该文件是ID3决策树算法的相关操作



from math import log
import operator


#计算给定数据集的香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob*log(prob,2)
    return shannonEnt

#按照给定特征划分数据集
#dataSet：待划分的数据集
#axis：划分数据集的特征--数据的第几列
#value：需要返回的特征值
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]           #获取从第0列到特征列的数据
            reducedFeatVec.extend(featVec[axis+1:])   #获取从特征列之后的数据
            retDataSet.append(reducedFeatVec)
    return retDataSet

#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0])-1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0;bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntroy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prop = len(subDataSet)/float(len(dataSet))
            newEntroy += prop * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntroy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

#该函数用于找出出现次数最多的分类名称
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classList.iteritems(), key=operator.itemgetter(1), reverse=True)   #利用operator操作键值排序字典
    return sortedClassCount[0][0]

#创建树的函数
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

#创建数据集
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
    return dataSet,labels