import numpy as np
import matplotlib.pyplot as plt
import copy

#设定训练数据集类
class Data():
    def __init__(self,data_input,data_output):
        self.data_input  = data_input
        self.data_output = data_output

    def is_positive(self,f):#输出f列表中所有大于0的数
        selected_points = []
        for i in range(0,row):
            if f[i]>0:#点被正确分类
                selected_points.append(f[i])#f1存储正确分类点
        return selected_points

    def train(self):
        flag1 = True

        # 设定w,b,学习率初值
        w = np.zeros(column)
        w1=[]
        b = 0
        learning = 1
        selected_points=[]#筛选后f1

        while flag1==True:#第二层循环，直到训练集中没有误分类点，退出
            model = [np.dot(w, data_input[0]) + b, np.dot(w, data_input[1]) + b, np.dot(w, data_input[2]) + b]
            f = [data_output[0] * model[0], data_output[1] * model[1], data_output[2] * model[2]]#矩阵A*B
            selected_points = self.is_positive(f)
            if len(selected_points) == row :#第二层循环推出条件，直到正确分类点数等于所有分类点数
                flag1 = False
            else:
                selected_points = []

                for i in range(0, row):#w,b参数更新，第一层循环，从x1到x3,找到一次遍历之后的w,b
                    flag2 = True
                    while flag2:
                            m = data_input[i]
                            f = np.dot(w, data_input[i]) + b
                            if data_output[i]*f <= 0:
                                w += learning * data_output[i] * data_input[i]
                                b += learning * data_output[i]
                                w=copy.deepcopy(w)#深度copy用于返回w


                                w1.append(w)
                            else:
                                flag2 = False

        return w,b,w1


if __name__ == '__main__':
    #定义输入输出

    data_input = np.array([[3,3],[4,3],[1,1]])#定义输入 创建[[多维数组]]注意 3*2
    data_output = [1,1,-1]#定义输出
    data = Data(data_input,data_output)
    row,column = np.shape(data.data_input)#遍历次数 #a.shape=[行，列]


    w,b,w1 = data.train()#算法

#画图

    plt.figure(figsize=(10, 5))
    x_values = [3, 4, 1]
    y_values = [3, 3, 1]
    plt.scatter(x_values, y_values, s=100)#画训练点

    for w3 in w1:
        print(w3)
        w0=w3[0]
        w1=w3[1]

        if w1!= 0:
            x=np.linspace(0,6,10)
            y=(-b-w0*x)/w1

            plt.plot(x,y)#画直线

    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False#实现中文输入
    plt.title('感知机学习算法的原始形式实现')

    plt.show()
