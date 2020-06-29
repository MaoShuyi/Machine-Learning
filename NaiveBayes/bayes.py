
"""
Created on 2020年3月31日20:00:51
Author: Shuyi Mao

朴素贝叶斯分类器 （连续值），
1. 连续值使用高斯分布计算概率
2. 原来以为只对整个训练集计算mean和std，其实是从训练集中分别提取各个类别的数据，单独对每个特征计算mean和std
   即class = 0 : mean&std.shape = [1, feature]
   最后总的mean&std.shaoe = [class_num, feature]
3. 计算概率时有 P(feature[i]|class=0), P(feature[i]|class=1) ……

原代码：https://github.com/htshinichi/ML_model/tree/master/NaiveBayes
自己跟着代码走一遍复写，其中取消了原代码中单个测试数据的预测，在PredTestSet()函数中直接使用矩阵运算，减少了一层test_num的循环
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class NaiveBayesContinuous:

    def __init__(self, train_set, test_set):
        label_count = train_data.label.value_counts()

        self.train_set = train_set                  # train_set（不太清楚需不需要self。。）
        self.labels = np.array(label_count.index)   # label_arr = [0,1]
        self.class_num = len(self.labels)           # class_num = 2  类别个数

        self.feature_num = train_data.columns.size - 1  # feature_num 特征维度

        self.test_set = np.array(test_set)          # 把dataframe -> np， 不知道dataframe切片操作和np的一不一样，需不需要转
        self.test_num = len(test_set)               # 测试样本个数


    def CalcMeanStd(self):
        """
        计算训练集每个类别下每个特征的mean和std
        ！！！ 循环计算时应该可以使用矩阵计算，减少一层循环和取消字典运算，但是现在没时间
        :return: mean,std shape: [class, feature]
        """
        d = {}
        mean = []
        std = []

        # 按class分类
        for i in range(self.class_num):
            d["train_{}".format(i)] = train_data[train_data["label"] == i]

        for i in range(self.class_num):
            d["c{}_mean".format(i)] = []
            d["c{}_std".format(i)] = []

            # 按特征维度计算每个class的mean和std
            for j in range(self.feature_num):
                d["c{}_mean".format(i)].append(np.mean(d["train_{}".format(i)]['%s' % j]))
                d["c{}_std".format(i)].append(np.std(d["train_{}".format(i)]['%s' % j], ddof=1))

            mean.append(d["c{}_mean".format(i)])
            std.append(d["c{}_std".format(i)])

        return mean, std


    def PredTestSet(self):
        """
        用于计算整个测试集，通过一层类别循环计算各个特征下的概率 及 一层特征循环对每个概率累乘 得到最后的总概率
        :return:
        """
        allProb = np.empty((self.test_num, self.class_num))

        mean, std = self.CalcMeanStd()
        for i in range(self.class_num):
            probs = self.GaussProb(self.test_set[:, 0:8], mean[i], std[i])
            prob_temp = 1

            for j in range(self.feature_num):       # 对所有的特征下的概率累乘
                prob_temp *= probs[:,j]

            allProb[:, i] = prob_temp               # 每个样本对应每个类别的概率 [test_num, class_num]

        pred_classes = np.argmax(allProb, axis=1)   # 按行找出概率最大的索引，即为预测的类别
        return pred_classes


    def Accuracy(self):
        pred_classes = np.array(self.PredTestSet()).reshape(-1,1)
        true_classes = self.test_set[:, -1].reshape(-1,1)

        mask = (pred_classes == true_classes)
        right_num = np.sum(mask != 0)

        accuracy = right_num / len(self.test_set) * 100
        print("准确率为：", accuracy, "%")
        return accuracy


    def GaussProb(self, x, mean, std):
        exponent = np.exp(-(np.power(x - mean, 2)) / (2 * np.power(std, 2)))
        GaussProb = (1 / (np.sqrt(2 * np.pi) * np.array(std))) * exponent
        return GaussProb


if __name__ == '__main__':

    datas = pd.read_csv("D:\\dataset\\pima\\pima-indians-diabetes.data.csv")
    train_data, test_data = train_test_split(datas, test_size=0.2)

    model_NB = NaiveBayesContinuous(train_data, test_data)
    accuracy = model_NB.Accuracy()
