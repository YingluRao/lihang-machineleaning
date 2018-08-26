import numpy as np
import csv
from collections import Counter
import json
def exp_entropy(dataSet):#计算经验熵
    num = len(dataSet)#m行，n列
    labelcounts = {}
    for datalabel in dataSet[:,-1]:
        if datalabel not in labelcounts.keys():
            labelcounts[datalabel] = 0
        labelcounts[datalabel] += 1#计算最后一列不同label的个数

    HD = 0
    for key in labelcounts:
        prob = float(labelcounts[key]) / num #总数据个数
        HD -= prob * np.log2(prob)
    return HD

def spiltdataSet(dataSet,axis,value):
    retdataset = []
    count = 0
    for data in dataSet:
        if data[axis]==value:
            retdataset.append(data)
            count = count + 1
    return retdataset


def choose_best_feature(dataSet,featureSet):
    baseHD = exp_entropy(dataSet)
    numfeature = len(featureSet)-1
    HD_all = []
    for i in range(numfeature):#4个特征
        HD_current = 0
        featurelist = [number[i] for number in dataSet]
        featurelist = set(featurelist)
        for value in featurelist:#3个特征 young ,middle,old
            subdataSet = spiltdataSet(dataSet,i,value)#第i个特征，第i个特征的值
            subdataSet = np.array(subdataSet)
            #print(np.array(subdataSet))#list转array
            HDI = exp_entropy(subdataSet)
            count = len(subdataSet)
            HD_current +=  HDI*count
        HD_all.append(baseHD-HD_current)
    index_best = HD_all.index(max(HD_all))
    return index_best

def create_tree(dataSet,featureSet):
    datarow,datacolomn = np.shape(dataSet)
    dataSet = np.array(dataSet)
    dataSet = dataSet.reshape(-1,datacolomn)
    if len(set(dataSet[:,-1]))==1:
        return dataSet[:,-1][0]#假如所有实例属于同一类，返回类型
    if len(featureSet)==1:
        return Counter(dataSet[:,-1]).most_common(1)[0][0]

    bestfeat = choose_best_feature(dataSet,featureSet)
    bestfeatlabel = featureSet[bestfeat]
    my_tree = {bestfeatlabel:{}}
    newfeatureSet = np.delete(featureSet,bestfeat,axis=0)#删除已检测过元素后

    featvalue = set(dataSet[:,bestfeat])
    for feat in featvalue:
        sublabels = newfeatureSet
        my_tree[bestfeatlabel][feat]=create_tree(spiltdataSet(dataSet,bestfeat,feat),sublabels)
        #多理解这一句怎么递归的
    return my_tree

def openfile():
    data = open('data.csv','r',encoding='utf-8')#不要用rb模式，返回的不是字符,标注encoding='utf-8'不然会报错乱码
    reader = csv.reader(data)
    #输入 训练数据集D，特征A 进行数据预处理
    dataSet = []
    for line in reader:#第一行
        dataSet.append(line)
    dataSet = np.array(dataSet)#list转矩阵
    featureSet = dataSet[:,1:][0]
    dataSet = dataSet[1:,1:]#提取dataset的第二行到最后一行，第二列到最后一列
    return dataSet,featureSet




if __name__ == '__main__':#只在我执行该文件时才运行，如果该模块被调用，不会执行，用于测试
    dataSet,featureSet = openfile()
    tree = create_tree(dataSet,featureSet)
    print(tree)
    print(json.dumps(tree, indent=1))