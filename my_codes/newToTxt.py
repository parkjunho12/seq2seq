# -*- coding: utf-8 -*-
file_path = "../data/news.csv"
import re

if __name__ == '__main__':

    import pandas as pd
    datas = pd.read_csv(file_path, engine='python', names=['원본 TITLE', '원본 CONTENT', '요약 TITLE', '요약 CONTENT'])
    origin_content = datas.loc[:, '원본 CONTENT'].values
    sum_content = datas.loc[:, '요약 CONTENT'].values
    origin_title = datas.loc[:, '원본 TITLE'].values
    sum_title = datas.loc[:, '요약 TITLE'].values
    index = 0

    f = open("../data/NEWS_SUM.txt", 'w',-1 ,"utf-8")
    print(len(sum_content))
    for content in sum_content:
        if content == "":
            continue
        index = index + 1
        print(content)
        contents = str(content).split('\n')
        text = "".join([s for s in str(content).splitlines(True) if s.strip("\r\n")])
        texts = text.splitlines()
        for t in texts:
            if index != len(sum_title):
                f.write(t)
        f.write('\n')
    f.close()
