#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''
Created on Oct 12, 2010
Update on 2021-04-07
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: NoneType/Mr.Q
《机器学习实战》更新地址：
'''

# print(__doc__)  ＃　上面的信息打印
import operator
from math import log

# collections.Counter( data )  返回的是一个字典
from collections import Counter

def calcShannonEnt(dataset):
    """
    计算熵值
    :param dataset:
    :return:
    """

    # -----------计算香农熵的第一种实现方式start--------------------------------------------------------------------------------
    numEntries = len(dataset)  # 获取数据集中的数量
    labelCounts = {}

    for featVec in dataset:
        currentLabel = featVec[-1]  # 只对最后一列--标签进行计算
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0  # 创建标签
        labelCounts[currentLabel] += 1

    shannonEnt = 0.0
    for key in labelCounts:
        # 使用所有类标签的发生频率计算类别出现的概率。
        prob = float(labelCounts[key]) / numEntries
        # 假定log底数为２
        shannonEnt -= prob * log(prob, 2)
        # print('---', prob, prob * log(prob, 2), shannonEnt)
    # -----------计算香农熵的第一种实现方式end--------------------------------------------------------------------------------

    # # -----------计算香农熵的第二种实现方式start--------------------------------------------------------------------------------
    # # 统计标签出现的次数
    # label_count = Counter(data[-1] for data in dataSet)
    # # 计算概率
    # probs = [p[1] / len(dataSet) for p in label_count.items()]
    # # 计算香农熵
    # shannonEnt = sum([-p * log(p, 2) for p in probs])
    # # -----------计算香农熵的第二种实现方式end--------------------------------------------------------------------------------

    return shannonEnt

def createDataSet():
    """
    Desc:
        创建数据集
    Args:
        无需传入参数
    Returns:
        返回数据集和对应的label标签
    """
    # dataSet 前两列是特征，最后一列对应的是每条数据对应的分类标签
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]

    # dataSet = [['yes'],
    #         ['yes'],
    #         ['no'],
    #         ['no'],
    #         ['no']]
    # labels  露出水面   脚蹼，注意：这里的labels是写的 dataSet 中特征的含义，并不是对应的分类标签或者说目标变量
    labels = ['no surfacing', 'flippers']
    # 返回
    return dataSet, labels

def add_label_category():
    mydata, labels = createDataSet()
    mydata[0][-1] = 'maybe'  # 将第一个数据的标签改为maybe
    print(mydata)
    print(calcShannonEnt(mydata))

def splitDataSet(dataSet, index, value):
    """
        Desc：
            划分数据集
            splitDataSet(通过遍历dataSet数据集，求出index对应的colnum列的值为value的行)
            就是依据index列进行分类，如果index列的数据等于 value的时候，就要将 index 划分到我们创建的新的数据集中
        Args:
            dataSet  -- 数据集                 待划分的数据集
            index -- 表示每一行的index列        划分数据集的特征
            value -- 表示index列对应的value值   需要返回的特征的值。
        Returns:
            index 列为 value 的数据集【该数据集需要排除index列】
        """
    # -----------切分数据集的第一种方式 start------------------------------------
    retDataSet = []
    for featVec in dataSet:
        if featVec[index] == value:
            # [:index]表示前index行，即若 index 为2，就是取 featVec 的前 index 行,也就是第0,1行
            reducedFeatVec = featVec[:index]
            reducedFeatVec.extend(featVec[index+1:])
            # [index+1:]表示从跳过 index 的 index+1行，取接下来的数据,所以这里没有选取第index行
            # 收集结果值 index列为value的行【该行需要排除index列】

            retDataSet.append(reducedFeatVec)    # 最终返回的结果
    # -----------切分数据集的第一种方式 end------------------------------------

    # # -----------切分数据集的第二种方式 start------------------------------------
    # retDataSet = [data[:index] + data[index + 1:] for data in dataSet for i, v in enumerate(data) if i == index and v == value]
    # # -----------切分数据集的第二种方式 end------------------------------------
    return retDataSet
    '''
            请百度查询一下： extend和append的区别
            list.append(object) 向列表中添加一个对象object
            list.extend(sequence) 把一个序列seq的内容添加到列表中
            1、使用append的时候，是将new_media看作一个对象，整体打包添加到music_media对象中。
            2、使用extend的时候，是将new_media看作一个序列，将这个序列和music_media序列合并，并放在其后面。
            result = []
            result.extend([1,2,3]) # [1, 2, 3]
            result.append([4,5,6]) # [1, 2, 3, [4, 5, 6]]
            result.extend([7,8,9])  # [1, 2, 3, [4, 5, 6], 7, 8, 9]   
    '''

# [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
# 选取mydata第0列的特征＝＝０的样本，然后将把这些样本的第０列去除后返回
# print(splitDataSet(mydata, 0, 0))  # [[1, 'no'], [1, 'no']]
# print(splitDataSet(mydata, 0, 1))  # [[1, 'yes'], [1, 'yes'], [0, 'no']]


def chooseBestFeatureToSplit(dataSet):
    """
    Desc:
        选择切分数据集的最佳特征
    Args:
        dataSet -- 需要切分的数据集
    Returns:
        bestFeature -- 切分数据集的最优的特征列
    """
    # -----------选择最优特征的第一种方式 start------------------------------------
    # 求第一行有多少列的 Feature, 最后一列是label列
    numFeatures = len(dataSet[0]) - 1
    # label的信息熵
    baseEntropy = calcShannonEnt(dataSet)
    # 最优的信息增益值, 和最优的Featurn编号
    bestInfoGain, bestFeature = 0.0, -1

    # 遍历所有的feature，计算其熵值
    for i in range(numFeatures):
        # 获取每一个实例的第i+1个feature，组成list集合
        featList = [example[i] for example in dataSet]
        # 获取剔重后的集合，使用set对list数据进行去重
        uniqueVals = set(featList)
        # 创建一个临时的信息熵
        newEntropy = 0.0
        # 遍历某一列的value集合，计算该列的信息熵

        # 含有这个类并且值为value
        # 遍历当前特征中的所有唯一属性值，对每个唯一属性值划分一次数据集，计算数据集的新熵值，并对所有唯一特征值得到的熵求和。
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
            # gain[信息增益]: 划分数据集前后的信息变化， 获取信息熵最大的值
            # 信息增益是熵的减少或者是数据无序度的减少。最后，比较所有特征中的信息增益，返回最好特征划分的索引值。
        infoGain = baseEntropy - newEntropy
        print('infoGain=', infoGain, 'bestFeature=', i, baseEntropy, newEntropy)

        # 信息增益越大越好
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

    # # -----------选择最优特征的第二种方式 start------------------------------------
    # # 计算初始香农熵
    # base_entropy = calcShannonEnt(dataSet)
    # best_info_gain = 0
    # best_feature = -1
    # # 遍历每一个特征
    # for i in range(len(dataSet[0]) - 1):
    #     # 对当前特征进行统计
    #     feature_count = Counter([data[i] for data in dataSet])
    #     # 计算分割后的香农熵
    #     这里的feature是一个属性，可能有多个取值，每个取值对应的是feature[1],这个是每种属性出现的频率
    #     new_entropy = sum(feature[1] / float(len(dataSet)) * calcShannonEnt(splitDataSet(dataSet, i, feature[0])) \
    #                    for feature in feature_count.items())
    #     # 更新值
    #     info_gain = base_entropy - new_entropy
    #     print('No. {0} feature info gain is {1:.3f}'.format(i, info_gain))
    #     if info_gain > best_info_gain:
    #         best_info_gain = info_gain
    #         best_feature = i
    # return best_feature
    # # -----------选择最优特征的第二种方式 end------------------------------------




if __name__ == '__main__':
    mydata, labels = createDataSet()
    print(mydata)


'''
#-----------------------------------------------Collection.Counter用法-----------------------------------------------------
# 可以传入list, dict, string, tuple
count_key = Counter(['a','a','b'])
# print(count_key)
count_key = Counter({'eggs':2, 'fruit':3})
# print(count_key)
count_key = Counter(cat = 4, dog=  8)
# print(count_key)

# 返回所有的元素
# count_key.elements() 返回的是迭代器
print(list(count_key.elements()))  

print(count_key.items())　# dict_items([('cat', 4), ('dog', 8)])
# 与上面等价，但是不是返回迭代器 

# 返回值
print(count_key.values())

# 返回键
print(count_key.keys())

# 返回最常见的ｋ个元素
print(count_key.most_common(1))

# 更新元素
count_key['cat'] = 5
print(count_key)

# 操作两个Counter
c = Counter(a=3, b=1)
d = Counter(b=2, c=1)
print(c+d)

print(d-c)
print(c-d)

# 求并集
print(c | d)

# 求交集
print(c & d)

#-----------------------------------------------Collection.Counter用法-----------------------------------------------------
'''
count_key = Counter(cat = 4, dog=  8)
print(count_key.items()) # 与上面等价，但是不是返回迭代器