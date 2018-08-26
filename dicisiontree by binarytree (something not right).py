#ID3算法
import csv
import numpy as np
from collections import Counter
import datahandle#文件名不要有空格，不然不好导入
import copy
class node():
    def __init__(self,data=None,dataset=None,lchild=None,rchild=None):
        self.data = data #label
        self.dataset = dataset#实例
        self.lchild = lchild
        self.rchild = rchild

def step3(tree,threshold,list1):#返回树
    gdas = datahandle.main(list1)
    #print(gdas)#gdas是二维
    b = gdas.index(max(gdas))#得到最大值索引 2
    #print(gdas[b][0])得到数值
    if gdas[b][0] < threshold:#如果Ag信息增益小于阈值，则置T为单节点树，并将D中实例数最大的类CK作为节点类标记并返回
        tree.data=(Counter(list1[1:][:, -1]).most_common(1))[0][0]
        tree.dataset=list1
        return tree
    else:
        tree.data=list1[0][b+1]
        tree.dataset=list1
        #print('class label:' + str(list1[0][b+1]))
        classlabel = set(list1[1:][:, b + 1])#yes,no
        classlabel = list(classlabel)
        m,n = np.shape(list1)
        listaa = list1[0]
        listbb = list1[0]
        for i in range(1,m):
            if list1[i][b+1]==classlabel[0]:
                listaa = np.hstack((listaa,list1[i]))
            else:
                listbb = np.hstack((listbb,list1[i]))

        listaa = np.reshape(listaa,(-1,n))
        listbb = np.reshape(listbb,(-1,n))#用于循环
        listaa1 = copy.deepcopy(listaa)
        listbb1 = copy.deepcopy(listbb)
        listaa1 = np.delete(listaa1,b+1,axis=1)
        listbb1 = np.delete(listbb1,b+1,axis=1)#用于计算gda


        tree.lchild=node(data = classlabel[0],dataset=listaa)
        tree.rchild=node(data=classlabel[1],dataset=listbb)
        step1and2(tree.lchild,listaa1)
        step1and2(tree.rchild,listbb1)
        return tree

def step1and2(tree,list1):
    #如果D中所有实例属于同一类CK，则T为单结点树，并将类CK作为该节点的类标记
    exampleclass = list1[:, -1].tolist()
    if set(exampleclass)=={'yes','class'}:#{yes}
        tree.data = 'yes'
        tree.dataset=list1
        return tree
    elif set(exampleclass)=={'no','class'}:
        tree.data = 'no'
        tree.dataset = list1
        return tree
    #如果A为空集，则T为单节点树，并将D中最大的类CK作为该节点的类标记，返回T
        #print(list1[0][1:-1])#所有特征，第一行为特征行，去掉首尾
    elif list1[1:-1]==[]:#如果列表为空
        tree.data = Counter(list1[:,-1]).most_common(1)[0][0]
        tree.dataset = list1#打印[('yes',9)],把yes给node
        return tree
    else:
        threshold = 0.1#阈值threshold=0.1
        tree = step3(tree,threshold,list1)
        return tree


def preorder(tree1):  # 先序遍历
    print(tree1.data)
    print(tree1.dataset)
    if tree1.lchild is not None:  # 节点不为空
        preorder(tree1.lchild)
    if tree1.rchild is not None:
        preorder(tree1.rchild)




if __name__ == '__main__':#只在我执行该文件时才运行，如果该模块被调用，不会执行，用于测试
    data = open('data.csv','r',encoding='utf-8')#不要用rb模式，返回的不是字符,标注encoding='utf-8'不然会报错乱码
    reader = csv.reader(data)
    #输入 训练数据集D，特征A 进行数据预处理
    list1 = []
    for line in reader:#第一行
        list1.append(line)#shape 14,6
    list1 = np.array(list1) #把reader中的文件转为list1（16，6）
    tree = node()
    tree1 = step1and2(tree,list1)
    preorder(tree1)





