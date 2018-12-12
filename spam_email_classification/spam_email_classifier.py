import os
import pickle
import re
import time

import jieba
import numpy as np
from sklearn import metrics, tree
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import KFold
from sklearn.svm import SVC


def get_stop_words(file_of_stop_words):  # 获取停用词表
    stop_words = []
    with open(file_of_stop_words, 'r', encoding="utf-8") as file:
        lines = file.readlines()  # txt中所有字符串读入data
        for line in lines:
            if len(line) == 0:
                continue
            stop_words.append(line)

    return stop_words


def remove_symbol(train_file_path, save_path):  # 去除非中文的字符
    all_data = []
    with open(train_file_path, 'r', encoding="utf-8") as file:
        data = file.readlines()  # txt中所有字符串读入data
        for line in data:
            label, content = int(line[0]), line[1:].strip()
            # content 里面去除无关的符号，只留下有意义的中文，并用逗号连接
            p2 = re.compile('[^\u4e00-\u9fa5]')  # 中文的编码范围是：\u4e00到\u9fa5
            zh = " ".join(p2.split(content)).strip()
            zh = ",".join(zh.split())
            content_seg = jieba.cut(zh)
            all_data.append([label, " ".join(content_seg)])
    # save the data to file
    dataset = dict(all_data=all_data)
    fpkl = open(save_path, 'wb')
    pickle.dump(dataset, fpkl, protocol=4)  # 将对象转化为文件保存在磁盘
    fpkl.close()
    return all_data


def update_with_tf_idf(train_data, test_data):  # 生成对应的TF-IDF矩阵，共享词汇
    transformer = TfidfTransformer()

    train_data = {"input": [i[1] for i in train_data], "label": [int(i[0]) for i in train_data]}
    test_data = {"input": [i[1] for i in test_data], "label": [int(i[0]) for i in test_data]}
    train_contents = train_data["input"]
    test_contents = test_data["input"]

    # train data vectorizer.fit_transform(contents)计算个词语出现的次数
    train_vectorizer = CountVectorizer(stop_words=get_stop_words("stopword/哈工大停用词表.txt"))
    # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    tfidf_train = transformer.fit_transform(train_vectorizer.fit_transform(train_contents))
    # print(np.array(tfidf_train).shape)

    # test data
    test_vectorizer = CountVectorizer(vocabulary=train_vectorizer.vocabulary_)
    tfidf_test = transformer.fit_transform(test_vectorizer.fit_transform(test_contents))

    train_data["input"] = tfidf_train
    test_data["input"] = tfidf_test
    return train_data, test_data


def read_cache(all_data_file):
    pkl_file = open(all_data_file, 'rb')
    all_data = pickle.load(pkl_file)["all_data"]  # [:10000]
    pkl_file.close()
    return all_data


def train_svm(train_data, test_data):
    svclf = SVC(kernel='linear')  # default with 'rbf'
    svclf.fit(train_data["input"], train_data["label"])
    pred = svclf.predict(test_data["input"])
    return pred


def train_dt(train_data, test_data):
    model = tree.DecisionTreeClassifier()
    model.fit(train_data["input"], train_data["label"])
    pred = model.predict(test_data["input"])
    return pred


if __name__ == '__main__':
    all_data_file = "data_labeled_all.tmp"
    if os.path.isfile(all_data_file):
        all_data = read_cache(all_data_file)
    else:
        all_data = remove_symbol(train_file_path="带标签短信.txt", save_path=all_data_file)

    all_data = np.array(all_data)
    kf = KFold(n_splits=5, shuffle=True, random_state=5)

    precision = []
    recall = []
    f1 = []
    since = time.time()
    for i, (train_index, test_index) in enumerate(kf.split(all_data)):
        # 第k折
        train_data, test_data = update_with_tf_idf(train_data=all_data[train_index], test_data=all_data[test_index])

        # build model and train it
        pred = train_svm(train_data, test_data)

        precision.append(metrics.precision_score(test_data["label"], pred))
        recall.append(metrics.recall_score(test_data["label"], pred))
        f1.append(metrics.f1_score(test_data["label"], pred))
        print(i, precision[-1], recall[-1], f1[-1])

    average_precision = np.mean(precision)
    average_recall = np.mean(recall)
    average_f1 = np.mean(f1)
    print("precision: {}\trecall: {}\tF1: {}\n".format(average_precision, average_recall, average_f1))
    time_elapsed = time.time() - since
    print('Training and prediction complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
