# -*- coding: utf-8 -*-
file_path = "./201912.csv"
import re

if __name__ == '__main__':

    import pandas as pd
    datas = pd.read_csv(file_path, engine='python'  ,names=['reply'])
    replies = datas.loc[:, 'reply'].values
    index = 0

    f = open("201912.txt", 'w')
    f.write("document\n")
    for reply in replies:
        hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')
        result = hangul.sub("", reply)
        if result == "":
            continue
        index = index + 1
        print(index)
        print(result)
        f.write(result+"\n")
    f.close()