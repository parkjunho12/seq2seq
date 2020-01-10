# -*- coding: utf-8 -*-
file_path = "./news.csv"
import re

if __name__ == '__main__':

    import pandas as pd
    datas = pd.read_csv(file_path, engine='python', names=['원본 TITLE', '원본 CONTENT', '요약 TITLE', '요약 CONTENT'])
    origin_content = datas.loc[:, '원본 CONTENT'].values
    sum_content = datas.loc[:, '요약 CONTENT'].values
    index = 0

    f = open("NEWS.txt", 'w',-1 ,"utf-8")
    for content in origin_content:
        if content == "":
            continue
        index = index + 1
        print(index)
        print(content)
        f.write(content+"\n다음 기사에요\n")
    f.close()