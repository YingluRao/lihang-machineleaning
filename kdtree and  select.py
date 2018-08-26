import numpy as np
import math
class Node():#建立节点
    def __init__(self,data,axis,lchild,rchild,root):
        self.data = data
        self.axis = axis
        self.lchild = lchild
        self.rchild = rchild
        self.root = None

def root(node1):#建立根节点
        if node1.lchild:
            node1.lchild.root=node1
        if node1.rchild:
            node1.rchild.root=node1

        if node1.lchild :
            root(node1.lchild)
        if node1.rchild:
            root(node1.rchild)


def CreateNode(axis,data):#按照kd平衡树规则建立树

    if not data:
        return None
    else:
        data = sorted(data, key=lambda x: x[axis])
        medium = len(data) // 2
        axisnew = (axis+1) % 2
        rootdata = data[medium]
        nodereturn = Node(rootdata,axis,CreateNode(axisnew,data[:medium]),CreateNode(axisnew,data[medium+1:]),root)
        return nodereturn

def preorder(root):#先序遍历
    print(root.data)
    if root.lchild is not None:  # 节点不为空
        preorder(root.lchild)
    if root.rchild is not None:
        preorder(root.rchild)

def find_leave(node1,axis,x_input):#利用递归找到包含输入点的叶节点
    axis = axis % len(node1.data)#axis%2
    if node1.lchild is not None or node1.rchild is not None:
        if x_input[axis] < node1.data[axis]:#注意是node1.data[axis]
            node1 = node1.lchild
            axis = axis+1
            find_leave(node1,axis,x_input)
        if x_input[axis] > node1.data[axis]:
            node1 = node1.rchild
            axis = axis+1
            find_leave(node1,axis,x_input)
    return node1

def distance(node):
    L1 = (node.data[0]-x_input[0])**2+(node.data[1]-x_input[1])**2
    L2 = math.sqrt(L1)#得到点与x的距离
    return L2

def find_nearest(node_current_near,x_input= [3,4.5]):
    L3 = distance(node_current_near)#得到当前最近点与X的距离
    if node_current_near.root is None:
        return node_current_near

    if node_current_near == node_current_near.root.lchild:
        node_current = node_current_near.root.rchild
        L4 = distance(node_current)
        if L3>L4:#如果当前最近点的根节点的另一个子节点距离x更近，那么将另一个子节点作为最新的最近点
            node_current_near = node_current
            find_nearest(node_current_near)
        else:
            node_current_near = node_current_near.root
            find_nearest(node_current_near)#如果当前最近点距离x更近，那么返回到上一级根节点
    elif node_current_near == node_current_near.root.rchild:
        node_current = node_current_near.root.lchild
        L4 = distance(node_current)
        if L3>L4:
            node_current_near = node_current
            find_nearest(node_current_near)
        else:
            node_current_near = node_current_near.root
            find_nearest(node_current_near)

    return node_current_near




if __name__ == '__main__':
    datainput = [[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]]
    node1=CreateNode(0,datainput)
    if node1:
        print(type(node1))
        preorder(node1)
    x_input = [3,4.5]
    node_leave = find_leave(node1,axis=0,x_input=[2,5])
    the_nearest = node_leave.data#纪录当前最近点

    root(node1)
    node2=find_nearest(node_leave,x_input)
    print('the nearest is: ' )
    print(node2.data)
    print(list(range(0,3)))