#!/usr/bin/env python
# -*- coding: UTF-8 -*-


# 代码来源（有修改）：http://jorbe.sinaapp.com/2014/07/13/decesion-tree-implement-with-python
# 该代码实现了决策树 C4.5算法分类
# 数据集为鸢尾花数据集

# 该代码未剪枝，从结果看属于过度拟合，测试的准确率略低


import sys
from math import log
import operator
from numpy import mean
import pickle



# 返回所有数据集labels(列表)
def get_labels(train_file):
    labels = []
    f = open(train_file, 'r')
    while(1):
        line = f.readline().strip()
        if not line:
            break
        label = line.split(',')[-1]
        labels.append(label)
    return labels


# 返回dataset(即数据的列表集合)和features(即特征的列表)
def format_data(dataset_file):
    #sepal length（花萼长度）、sepal width（花萼宽度）、petal length（花瓣长度）、petal width（花瓣宽度）
    features = ['sepal_length','sepal_width','petal_length','petal_width']
    dataset = []
    f = open(dataset_file, 'r')
    while(1):
        line = f.readline().strip()
        if not line:
            break
        fea_and_label = line.split(',')
        dataset.append([float(fea_and_label[i]) for i in range(len(fea_and_label)-1)]+[fea_and_label[len(fea_and_label)-1]])
    return dataset, features


# 按指定feature划分数据集，返回四个列表:
# dataset_less:指定特征项的属性值＜=该特征项平均值的子数据集
# dataset_greater:指定特征项的属性值＞该特征项平均值的子数据集
# label_less:按指定特征项的属性值＜=该特征项平均值切割后子标签集
# label_greater:按指定特征项的属性值＞该特征项平均值切割后子标签集
def split_dataset(dataset, feature_index, labels):
    dataset_less = []
    dataset_greater = []
    label_less = []
    label_greater = []
    datasets = []
    for data in dataset:
        datasets.append(data[0:4])
    mean_value = mean(datasets, axis = 0)[feature_index]   #数据集在该特征项的所有取值的平均值
    for data in dataset:
         if data[feature_index] > mean_value:
             dataset_greater.append(data)
             label_greater.append(data[-1])
         else:
             dataset_less.append(data)
             label_less.append(data[-1])
    return dataset_less, dataset_greater, label_less, label_greater


# 计算数据集的熵大小
def cal_entropy(dataset):
    n = len(dataset)
    label_count = {}
    for data in dataset:
        label = data[-1]
        if label_count.has_key(label):
            label_count[label] += 1
        else:
            label_count[label] = 1
    entropy = 0
    for label in label_count:
        prob = float(label_count[label])/n
        entropy -= prob * log(prob,2)
    #print 'entropy:', entropy
    return entropy


# 计算指定特征对数据集的信息熵
def cal_info(dataset, feature_index):
    datasets = []
    for data in dataset:
        datasets.append(data[0:4])
    mean_value = mean(datasets, axis = 0)[feature_index]    #计算指定特征的所有数据集值的平均值
    #print mean_value
    dataset_less = []
    dataset_greater = []
    for data in dataset:
        if data[feature_index] > mean_value:
            dataset_greater.append(data)
        else:
            dataset_less.append(data)
    condition_entropy = float(len(dataset_less))/len(dataset)*cal_entropy(dataset_less) + float(len(dataset_greater))/len(dataset)*cal_entropy(dataset_greater)
    return condition_entropy


# 计算信息增益比
# ！此处是与ID3算法的区别，通过比较比率，使算法更合理
# ！此处有修改，便于理解
def cal_info_gain_ratio(dataset, feature_index):
    base_entropy = cal_entropy(dataset)
    info = cal_info(dataset, feature_index)
    info_gain = base_entropy - info
    if info == 0:
        info_gain_ratio = 1
    else:
        info_gain_ratio = info_gain/info
    return info_gain_ratio


# 根据每个特征的信息增益比大小，返回最佳划分数据集的特征索引
def choose_best_fea_to_split(dataset, features):
    base_entropy = cal_entropy(dataset)
    split_fea_index = -1
    max_info_gain_ratio = 0.0
    for i in range(len(features)):
        info_gain_ratio = cal_info_gain_ratio(dataset, i)
        if info_gain_ratio > max_info_gain_ratio:
            max_info_gain_ratio = info_gain_ratio
            split_fea_index = i
    return split_fea_index


# 返回数据集中出现次数最多的label
def most_occur_label(labels):
    label_count = {}
    for label in labels:
        if label not in label_count.keys():
            label_count[label] = 1
        else:
            label_count[label] += 1
    sorted_label_count = sorted(label_count.iteritems(), key = operator.itemgetter(1), reverse = True)
    return sorted_label_count[0][0]


# 递归地创建决策树
# dataset:训练数据集
# labels:数据集中包含的所有label(可重复)
# features:可进行划分的特征集
def build_tree(dataset, labels, features, threshold):
    #若数据集为空,返回NULL
    if len(labels) == 0:
        return 'NULL'
    #若数据集中只有一种label,返回该label
    if len(labels) == len(labels[0]):
        return labels[0]
    #若没有可划分的特征集,则返回数据集中出现次数最多的label
    if len(features) == 0:
        return most_occur_label(labels)
    #若数据集趋于稳定，则返回数据集中出现次数最多的label
    if cal_entropy(dataset) == 0:
        return most_occur_label(labels)
    split_feature_index = choose_best_fea_to_split(dataset, features)
    split_feature = features[split_feature_index]
    decesion_tree = {split_feature:{}}
    #若划分特征的信息增益比小于阈值,则返回数据集中出现次数最多的label
    # ！此处阈值为0.3
    if cal_info_gain_ratio(dataset, split_feature_index) < threshold:
        return most_occur_label(labels)
    del(features[split_feature_index])
    dataset_less, dataset_greater, labels_less, labels_greater = split_dataset(dataset, split_feature_index, labels)
    decesion_tree[split_feature]['<='] = build_tree(dataset_less, labels_less, features, threshold)
    decesion_tree[split_feature]['>'] = build_tree(dataset_greater, labels_greater, features, threshold)
    return decesion_tree


# 把决策树以二进制格式写入文件
def store_tree(decesion_tree, filename):
    writer = open(filename, 'w')
    pickle.dump(decesion_tree, writer)
    writer.close()


# 从文件中读取决策树，返回决策树
def read_tree(filename):
    reader = open(filename, 'r')
    return pickle.load(reader)


# 对测试数据进行分类
def classify(decesion_tree, features, test_data, mean_values):
    first_fea = decesion_tree.keys()[0]
    fea_index = features.index(first_fea)
    if test_data[fea_index] <= mean_values[fea_index]:
        sub_tree = decesion_tree[first_fea]['<=']
        if type(sub_tree) == dict:
            return classify(sub_tree, features, test_data, mean_values)
        else:
            return sub_tree
    else:
        sub_tree = decesion_tree[first_fea]['>']
        if type(sub_tree) == dict:
            return classify(sub_tree, features, test_data, mean_values)
        else:
            return sub_tree


# 获取训练数据集各个属性的数据平均值
def get_means(train_dataset):
    dataset = []
    for data in train_dataset:
        dataset.append(data[0:4])
    mean_values = mean(dataset, axis = 0)   #数据集在该特征项的所有取值的平均值
    return mean_values


# 主函数
def run(train_file, test_file, threshold):
    labels = get_labels(train_file)
    train_dataset, train_features = format_data(train_file)
    decesion_tree = build_tree(train_dataset, labels, train_features, threshold)
    print 'decesion_tree :',decesion_tree
    store_tree(decesion_tree,'decesion_tree')

    mean_values = get_means(train_dataset)
    test_dataset, test_features = format_data(test_file)
    n = len(test_dataset)
    correct = 0
    for test_data in test_dataset:
        label = classify(decesion_tree, test_features, test_data, mean_values)
        if label == test_data[-1]:
            correct += 1
    print "准确率: ".decode('utf8'), correct/float(n)



if __name__ == '__main__':
    run("irisTrain.txt", "irisTest.txt", 0.3)   #阈值为0.3
