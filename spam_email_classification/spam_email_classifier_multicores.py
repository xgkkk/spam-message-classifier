import os
import pickle
import re
import time
from argparse import ArgumentParser
from multiprocessing import Manager
from multiprocessing.pool import Pool

import jieba
import numpy as np
from sklearn import metrics, tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import KFold, GridSearchCV
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


def read_test_data(test_file_path, words_file):  # 去除非中文的字符
    all_data = []
    all_data_ori = []
    with open(test_file_path, 'r', encoding="utf-8") as file:
        data = file.readlines()  # txt中所有字符串读入data
        for line in data:
            content = line.strip()
            all_data_ori.append(content)
            # content 里面去除无关的符号，只留下有意义的中文，并用逗号连接
            p2 = re.compile('[^\u4e00-\u9fa5]')  # 中文的编码范围是：\u4e00到\u9fa5
            zh = " ".join(p2.split(content)).strip()
            zh = ",".join(zh.split())
            content_seg = jieba.cut(zh)
            all_data.append(" ".join(content_seg))

    # test data
    TfidfVectorizer()
    pkl_file = open(words_file, 'rb')
    VOCABULARY = pickle.load(pkl_file)["words"]
    pkl_file.close()

    VECTORIZER = CountVectorizer(vocabulary=VOCABULARY)
    transformer = TfidfTransformer()

    tfidf_test = transformer.fit_transform(VECTORIZER.transform(all_data))
    return all_data_ori, tfidf_test


def update_with_tf_idf(train_data, test_data, save_path="words.tmp"):  # 生成对应的TF-IDF矩阵，共享词汇
    transformer = TfidfTransformer()

    train_data = {"input": [i[1] for i in train_data], "label": [int(i[0]) for i in train_data]}
    test_data = {"input": [i[1] for i in test_data], "label": [int(i[0]) for i in test_data]}
    train_contents = train_data["input"]
    test_contents = test_data["input"]

    # train data vectorizer.fit_transform(contents)计算个词语出现的次数
    train_vectorizer = CountVectorizer(stop_words=get_stop_words("stopword/哈工大停用词表.txt"))
    # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    tfidf_train = transformer.fit_transform(train_vectorizer.fit_transform(train_contents))
    # print(tfidf_train.shape)
    # test data
    test_vectorizer = CountVectorizer(vocabulary=train_vectorizer.vocabulary_)
    tfidf_test = transformer.fit_transform(test_vectorizer.fit_transform(test_contents))

    # if not os.path.isfile(save_path):
    dataset = dict(words=train_vectorizer.vocabulary_)
    fpkl = open(save_path, 'wb')
    pickle.dump(dataset, fpkl)  # 将对象转化为文件保存在磁盘
    fpkl.close()

    train_data["input"] = tfidf_train
    test_data["input"] = tfidf_test
    return train_data, test_data


def read_cache(all_data_file):
    pkl_file = open(all_data_file, 'rb')
    all_data = pickle.load(pkl_file)["all_data"]  # [:1000]
    pkl_file.close()
    return all_data


def train_model(info_dict, args_dict):
    train_data = info_dict["train_data"]
    test_data = info_dict["test_data"]
    model_name = args_dict["model_name"]
    if model_name == "svm":
        model = SVC(kernel='linear', probability=True)  # default with 'rbf'
    elif model_name == "dt":
        model = tree.DecisionTreeClassifier()
    elif model_name == "gbdt":
        model = GradientBoostingClassifier(learning_rate=args_dict["learning_rate"], max_features=args_dict["max_features"],
                                           max_leaf_nodes=args_dict["max_leaf_nodes"], max_depth=args_dict["max_depth"],
                                           n_estimators=args_dict["n_estimators"], random_state=10)

    model.fit(train_data["input"], train_data["label"])
    pred_train = model.predict(train_data["input"])
    pred_test = model.predict(test_data["input"])

    info_dict["train_precision"].append(metrics.precision_score(train_data["label"], pred_train))
    info_dict["train_recall"].append(metrics.recall_score(train_data["label"], pred_train))
    info_dict["train_f1"].append(metrics.f1_score(train_data["label"], pred_train))
    info_dict["precision"].append(metrics.precision_score(test_data["label"], pred_test))
    info_dict["recall"].append(metrics.recall_score(test_data["label"], pred_test))
    info_dict["f1"].append(metrics.f1_score(test_data["label"], pred_test))
    # print(info_dict["train_precision"][-1], info_dict["train_recall"][-1], info_dict["train_f1"][-1])
    print(info_dict["precision"][-1], info_dict["recall"][-1], info_dict["f1"][-1])
    dataset = dict(model=model)
    fpkl = open("{}/model_{}.tmp".format(model_name, info_dict["fold"]), 'wb')
    pickle.dump(dataset, fpkl)  # 将对象转化为文件保存在磁盘
    fpkl.close()
    return model


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, default="dt")
    parser.add_argument("-l", "--learning_rate", type=float, default=0.3)
    parser.add_argument("--max_features", type=str, default="sqrt")
    parser.add_argument("--max_leaf_nodes", type=int, default=50)
    parser.add_argument("--max_depth", type=int, default=50)
    parser.add_argument("-n", "--n_estimators", type=int, default=1400)
    args = parser.parse_args()
    model_name = args.model_name
    learning_rate = args.learning_rate
    max_features = args.max_features
    max_leaf_nodes = args.max_leaf_nodes
    max_depth = args.max_depth
    n_estimators = args.n_estimators

    if not os.path.isdir(model_name):
        os.mkdir(model_name)

    all_data_file = "data_labeled_all.tmp"
    if os.path.isfile(all_data_file):
        all_data = read_cache(all_data_file)
    else:
        all_data = remove_symbol(train_file_path="带标签短信.txt", save_path=all_data_file)

    all_data = np.array(all_data)
    kf = KFold(n_splits=5, shuffle=True, random_state=5)

    p = Pool(5)
    info_dict = Manager().dict()
    args_dict = Manager().dict()
    train_precision = Manager().list([])  # 主进程与子进程共享这个List
    train_recall = Manager().list([])  # 主进程与子进程共享这个List
    train_f1 = Manager().list([])  # 主进程与子进程共享这个List
    precision = Manager().list([])  # 主进程与子进程共享这个List
    recall = Manager().list([])  # 主进程与子进程共享这个List
    f1 = Manager().list([])  # 主进程与子进程共享这个List
    since = time.time()
    model_list = []
    for i, (train_index, test_index) in enumerate(kf.split(all_data)):
        # 第k折
        print(i)
        train_data, test_data = update_with_tf_idf(train_data=all_data[train_index], test_data=all_data[test_index],
                                                   save_path="{}\words_{}.tmp".format(model_name, i))
        # build model and train it
        info_dict = dict(train_data=train_data, test_data=test_data,
                         train_precision=train_precision, train_recall=train_recall, train_f1=train_f1,
                         precision=precision, recall=recall, f1=f1, fold=i)
        args_dict = dict(model_name=model_name, learning_rate=learning_rate, max_features=max_features, max_leaf_nodes=max_leaf_nodes,
                         max_depth=max_depth, n_estimators=n_estimators)
        model_list.append(p.apply_async(train_model, args=(info_dict, args_dict)))

    p.close()
    p.join()

    train_precision = np.mean(info_dict["train_precision"])
    train_recall = np.mean(info_dict["train_recall"])
    train_f1 = np.mean(info_dict["train_f1"])
    average_precision = np.mean(info_dict["precision"])
    average_recall = np.mean(info_dict["recall"])
    average_f1 = np.mean(info_dict["f1"])
    print("train: precision: {}\trecall: {}\tF1: {}\n".format(train_precision, train_recall, train_f1))
    print("test:  precision: {}\trecall: {}\tF1: {}\n".format(average_precision, average_recall, average_f1))
    time_elapsed = time.time() - since
    print('Training and prediction complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    best_f1_index = info_dict["f1"].index(max(info_dict["f1"]))
    print(best_f1_index)

    for _, i in enumerate(model_list):
        if _ != best_f1_index:
            continue
        test_model = i.get()

    test_data_ori, test_tfidf = read_test_data(test_file_path="不带标签短信.txt", words_file="{}/words_{}.tmp".format(model_name, best_f1_index))
    test_label = test_model.predict_proba(test_tfidf)

    txtfile = open("00_predict_result_{}.txt".format(model_name), 'w+', encoding="utf-8")
    for i in zip(test_label, test_data_ori):
        txtfile.write("{}\t{}\n".format(i[0], i[1]))
    txtfile.close()

    # write to file
    current = time.localtime()
    # current_date_time = time.strftime("%Y%m%d_%H%M%S", current)
    txtfile = open("result_{}.txt".format(model_name), 'a+')
    txtfile.write("{}\n".format("--" * 10))
    if model_name == "gbdt":
        txtfile.write("learning_rate: {}\tn_estimators: {}\tmax_depth: {}\tmax_features: {}\tmax_leaf_nodes: "
                      "{}\n".format(learning_rate, n_estimators, max_depth,
                             max_features, max_leaf_nodes))

    txtfile.write(
        "model name:{}\tcost time:{:.0f}m {:.0f}s\n".format(model_name, time_elapsed // 60, time_elapsed % 60))
    txtfile.write("(train)precision: {}\trecall: {}\tF1: {}\n".format(train_precision, train_recall, train_f1))
    txtfile.write("(test)precision: {}\trecall: {}\tF1: {}\n".format(average_precision, average_recall, average_f1))
    txtfile.close()
