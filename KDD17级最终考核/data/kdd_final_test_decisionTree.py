import numpy as np
import pandas as pd
from collections import Counter

data = pd.read_csv('./train.csv')


def data_process(data, rate, lable_pos=None):
    """
    数据处理函数
    :param data:
    :param rate:
    :param lable_pos:
    :return:
    """
    nparr = np.array(data)
    row = nparr.shape[0]
    train_set_num = int(row * rate)
    test_set_num = row - train_set_num  # just calculate,temperately don't use

    # 打乱数据顺序获得更好的耦合能力
    np.random.shuffle(nparr)

    train_set = nparr[0:train_set_num:, ]
    test_set = nparr[train_set_num::, ]
    if lable_pos is None:
        return train_set, test_set
    else:
        train_set_y = train_set[lable_pos]
        train_set_X = np.delete(train_set, lable_pos, 1)
        test_set_y = test_set[lable_pos]
        test_set_X = np.delete(test_set, lable_pos, 1)
        return train_set_X, train_set_y, test_set_X, test_set_y


def splite(X, y, axis, value):
    """
    分类函数
    :param X: 训练集的属性值
    :param y: 训练集的结果值
    :param axis: 维度
    :param value: 划分值
    :return: 返回的是训练集的两个分类子树
    一个子树包含X和y，所以共返回四个序列
    """
    index_a = (X[:, axis] <= value)
    index_b = (X[:, axis] > value)
    return X[index_a], X[index_b], y[index_a], y[index_b]


def calc_entropy(y):
    """
    计算信息熵
    :param y: 根据y来计算信息熵
    :return: 返回信息熵
    """
    counter = Counter(y)
    res = 0.0
    for num in counter.values():
        p = num / len(y)
        res += -p * np.log(y)
    return res


def find_best_node(X, y):
    """
    寻找最佳分类结点，使得信息熵最小
    :param X:训练集的属性
    :param y:训练集的结果
    :return:返回最佳节点
    """

    # 初始化最佳值
    best_entropy = float('inf')
    best_node, best_value = -1, -1

    for node in range(X.shape[1]):  # 对每一属性（node）进行遍历
        sorted_index = np.argsort(X[:, node])  # node列进行索引排序
        for i in range(1, len(X)):  # 对每一属性值（value）进行遍历
            if X[sorted_index[i - 1], node] != X[sorted_index[i], node]:
                value = (X[sorted_index[i - 1], node] + X[sorted_index[i], node]) / 2
                X_l, X_r, y_l, y_r = splite(X, y, node, value)
                e = calc_entropy(y_l) + calc_entropy(y_r)
                if e < best_entropy:
                    best_entropy = e
                    best_node = node
                    best_value = value
    return best_entropy, best_node, best_value


def create_tree(dataSet):
    end = dataSet.columns.size
    X = dataSet.ix[:, [i for i in range(1, end)]]
    y = dataSet.ix[:, [0]]
    _, best_node, best_value = find_best_node(X, y)
    splite(X, y, best_node, best_value)
    my_tree = {best_value: {}}
    return my_tree


class Node:
    def __init__(self, lft_child=None, lft_value=None, rgt_child=None, rgt_value=None):
        self.lft_child = lft_child
        self.lft_value = lft_value
        self.rgt_child = rgt_child
        self.rgt_value = rgt_value

    def add_node(self, layers=2):
        pass

