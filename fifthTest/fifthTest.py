#!/usr/bin/env python
# -*- coding: UTF-8 -*-


# 该代码未CART回归树
# 代码来源：机器学习实战（Peter Harrington 著） P159-165
# 该代码实现了CART算法，用于回归
# 回归树与分类树类似，但叶节点的数据类型不是分散型，而是连续型（该例中为均值）
# 该代码未剪枝


from numpy import *


# 读取数据集，并将每行映射为浮点数
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(',')
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    return dataMat


# 在给定特征和特征值的情况下，通过数组过滤的方式将数据集切分得到的两个子集返回
# P162有讲解，主要涉及到numpy模块的函数
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:][0]
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:][0]
    return mat0, mat1


# 生成叶节点
# 在回归树中，此未目标变量的均值
def regLeaf(dataSet):
    return mean(dataSet[:,-1])


# 误差估计函数，求出目标变量的平方误差
def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]


# 找到数据的最佳二元切分方式
# tolS是容许的误差下降值，tolN是切分的最少样本数
# leafType、errType两个参数指向函数
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0]
    tolN = ops[1]
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)    #如果所有值相等，则退出
    m, n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf    #正无穷
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:,featIndex]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1[0] < tolN)):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:
        return None, leafType(dataSet)     #如果误差减少不大，则退出
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)    #如果切分出的数据集很小，则退出
    return bestIndex, bestValue


# 递归地构建树
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree



if __name__ == '__main__':
    testMat = mat(eye(4))
    print testMat

    mat0, mat1 = binSplitDataSet(testMat, 1, 0.5)
    print mat0
    print mat1

    myDat = loadDataSet('ex00.txt')   #此处为训练文本
    myMat = mat(myDat)
    print createTree(myMat)