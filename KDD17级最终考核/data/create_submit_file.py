import os


def create_submit_file(list, file_name="submit_file"):
    """
    创建一个提交的文件函数
    :param list: 预测集列表
    :param file_name: 生成的文件名
    :return:
    """
    file_name = '{0}/{1}.{2}'.format(os.getcwd(), file_name, 'txt')
    while os.path.exists(file_name):
        file_name = '{0}/{1}.{2}'.format(os.getcwd(), file_name + '(1)', 'txt')
    with open(file_name, 'w+') as f:
        for i in range(len(list)):
            f.write(list[i] + '\n')
