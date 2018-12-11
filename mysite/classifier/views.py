import re
import time

import jieba
from django.conf import settings
from django.shortcuts import render


# Create your views here.


def index(request):
    if request.method == 'POST':
        email_content = request.POST["email_content"].strip()

        since = time.time()

        # content 里面去除无关的符号，只留下有意义的中文，并用逗号连接
        all_data = []
        p2 = re.compile('[^\u4e00-\u9fa5]')  # 中文的编码范围是：\u4e00到\u9fa5
        zh = " ".join(p2.split(email_content)).strip()
        zh = ",".join(zh.split())
        content_seg = jieba.cut(zh)
        all_data.append(" ".join(content_seg))

        model = settings.PRETRAINED_MODEL
        test_vectorizer = settings.VECTORIZER

        # TF-IDF
        tfidf_test = settings.TRANSFORMER.fit_transform(test_vectorizer.fit_transform(all_data))
        # a = model.predict_proba(tfidf_test)
        # b = model.predict(tfidf_test)

        since2 = time.time()
        results = model.predict_proba(tfidf_test)[0]
        time_elapsed = time.time() - since
        time_elapsed2 = time.time() - since2
        # print('Training and prediction complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        txtfile = open("result_online.txt", 'a+')
        txtfile.write("cost time:{:.3f}s\t{:.3f}s\n".format(time_elapsed % 60, time_elapsed2 % 60))
        txtfile.close()

        # print(results)
        notspam_p = round(results[0], 2)
        spam_p = round(results[1], 2)
        if notspam_p >= spam_p:
            spam_if = "不是"
        else:
            spam_if = "是"
        context = {"email_content": email_content, "click_if": 1, "spam_p": spam_p, "notspam_p": notspam_p, "spam_if": spam_if}
        return render(request, 'classifier/home.html', context)
    # form = InputForm()
    # print("testing")
    return render(request, 'classifier/home.html')

