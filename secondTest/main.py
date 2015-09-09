#!/usr/bin/env python
# -*- coding: UTF-8 -*-


# 代码来源：http://www.cnblogs.com/hantan2008/archive/2015/07/27/4674097.html
# 该代码实现了决策树算法分类（ID3算法）



import trees
import treePlotter

if __name__ == '__main__':
    pass

myDat, labels = trees.createDataSetFromTXT("dataset.txt")

shan = trees.calcShannonEnt(myDat)
print shan

col = trees.chooseBestFeatureToSplit(myDat)
print col

Tree = trees.createTree(myDat, labels)
print Tree

treePlotter.createPlot(Tree)