import os
import pickle  # 用于保存对象文件
import numpy as np
import pandas as pd
from collections import Counter


class Node:
    """
    这个Node类用来创建决策树
    """
    def __init__(self, lft_child=None, lft_value=None, rgt_child=None):
        self.lft_child = lft_child
        self.lft_value = lft_value
        self.rgt_child = rgt_child


def data_process(data, rate, lable_pos=None):
    """
    数据处理函数，形如Sklearn模型中的数据分离
    :param data: 数据集（csv文件通过pandas的read_csv函数创建一个DataFrame对象）
    :param rate: 数据留存率
    :param lable_pos: 标签值的位置
    :return: 返回两个数据集（训练集和测试集）
    """
    nparr = np.array(data)
    row = nparr.shape[0]
    train_set_num = int(row * rate)  # 计算出所需要的行数
    test_set_num = row - train_set_num  # just calculate,temperately don't use

    # 打乱数据顺序获得更好的耦合能力
    np.random.shuffle(nparr)

    train_set = nparr[0:train_set_num:, ]
    test_set = nparr[train_set_num::, ]

    if lable_pos is None:
        return train_set, test_set
    else:
        train_set_y = train_set[:, lable_pos]
        train_set_X = np.delete(train_set, lable_pos, 1)  # 分出标签值以后在属性集中就删除该列
        test_set_y = test_set[:, lable_pos]
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
    # 根据布尔掩玛来索引返回的分类集
    return X[index_a], X[index_b], y[index_a], y[index_b]


def calc_entropy(y):
    """
    计算信息熵
    :param y: 根据y来计算信息熵
    :return: 返回信息熵
    """
    counter = Counter(y)  # 计算出相同数据出现的次数，用以计算香农熵
    res = 0.0
    for num in counter.values():
        p = num / len(y)
        res += -p * np.log(p)  # 香农熵 E = - p * log(p)用来表示一个数据集的混乱度，在（0，1）内越小数据集越准确
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
        for i in range(1, X.shape[0]):  # 对每一属性值（value）进行遍历
            if X[sorted_index[i - 1], node] != X[sorted_index[i], node]:
                value = (X[sorted_index[i - 1], node] + X[sorted_index[i], node]) / 2
                X_l, X_r, y_l, y_r = splite(X, y, node, value)
                e = calc_entropy(y_l) + calc_entropy(y_r)
                if e < best_entropy:
                    best_entropy = e
                    best_node = node
                    best_value = value
    return best_entropy, best_node, best_value


def add_node(root, data_set_X, data_set_y):
    """
    增加节点函数，之后通过不断调用此函数来创建一颗决策二叉树
    :param root:输入根结点
    :param data_set_X: 数据的属性
    :param data_set_y: 数据的标签（结果）
    :return:
    """
    _, best_node, best_value = find_best_node(data_set_X, data_set_y)
    _, train_data_set_X, _, train_data_set_y = splite(data_set_X, data_set_y, best_node, best_value)  # 找到最佳节点之后要分割原数据集
                                                                                                      # 用作下次增加节点（分类）的数据
    root.lft_child = data.columns[best_node], best_node
    root.lft_value = best_value
    root.rgt_child = Node()  # 创建新的右节点
    return train_data_set_X, train_data_set_y  # 返回新的右子数据


def create_tree(root, data_set_X, data_set_y, layer=5):
    """
    创建决策树
    :param root: 根结点
    :param data_set_X: 数据的属性值
    :param data_set_y: 数据的标签值
    :param layer: 循环的层数，默认为5，即划分五次，分出六类
    :return:
    """
    for i in range(layer):
        data_set_X, data_set_y = add_node(root, data_set_X, data_set_y)
        root = root.rgt_child
    return


def predict(test_data_set_X, root):
    """
    根据测试数据集的属性值和决策树来预测测试数据的标签值（结果）
    :param test_data_set_X: 测试数据值
    :param root: 根结点（循环为数）
    :return: 返回预测值
    """
    predict_data = []
    for i in range(len(test_data_set_X)):
        while root.rgt_child is not None:
            if test_data_set_X[i, root.lft_child[1]] < root.lft_value:  # root.lft_child保存了列索引的名称和列()
                predict_data.append(root.lft_child[0])
            root = root.rgt_child
    return predict_data


def judge(predict_data, test_data_set_y):
    """
    判断一颗决策树性能的好坏，通过预测值正确的比例来判断
    :param predict_data: 预测值
    :param test_data_set_y: 精确值
    :return: 返回准确率
    """
    correct_num = 0
    test_data_set_y.reshape(len(predict_data))
    for i in range(len(predict_data)):
        if predict_data[i] == test_data_set_y[i]:
            correct_num += 1
    accurate_rate = correct_num / len(predict_data)
    return accurate_rate


def save_the_decision_tree(root, path, name="decision_tree_mode"):
    """
    通过创建好的决策树的根结点来保存一棵决策树模型至对应的目录
    :param root: 决策树根结点
    :param path: 目录的路径
    :return:
    """
    print("---正在保存决策树模型---")
    file_path = '{0}/{1}.{2}'.format(os.getcwd()+path, name, 'pkl')
    while os.path.exists(file_path):
        file_path = '{0}/{1}.{2}'.format(os.getcwd()+path, name+'(1)', 'pkl')
    with open(file_path, 'wb+') as f:
        pickle.dump(root, f)
    return


def load_a_decision_tree(file_path):
    with open(file_path, 'rb') as f:
        root = pickle.load(file_path)
    return root


if __name__ == "__main__":
    root = Node()
    data = pd.read_csv('./train.csv')
    train_data_set_X, train_data_set_y, test_data_set_X, test_data_set_y = data_process(data, 0.7, 0)  # 70%的数据用作训练，数据标签值为第一列
    create_tree(root, train_data_set_X, train_data_set_y)
    predict_data = predict(test_data_set_X, root)
    print("精确度为：", judge(predict_data, test_data_set_y) * 100, "%")
