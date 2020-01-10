# -*- coding: utf-8 -*-
file_path = "./books.csv"
import re

if __name__ == '__main__':

    import pandas as pd
    datas = pd.read_csv(file_path, engine='python', names=['Temp', '한국어', '영어', '영어 검수'])
    kor = datas.loc[:, '한국어'].values
    eng = datas.loc[:, '영어 검수'].values
    index = 0
    f = open("./result.txt", 'w',-1 ,"utf-8")
    for content in eng:
        if content == "":
            continue
        print(index)
        print(content)
        f.write(content + "\t" + kor[index] + "\n")
        index += 1
    f.close()