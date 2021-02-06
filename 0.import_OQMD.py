import pymysql
import pandas as pd
import numpy as np
import os
import xlrd

#导入OQMD数据库
con = pymysql.connect(host = "localhost", 
           user = "root", password = '123456', 
           db = "qmdb", charset='utf8')
sql = "select * from structures;"
data_raw = pd.read_sql(sql, con)
con.close()

#从数据库中搜索NbO数据
data = data_raw[data_raw.composition_id=='Nb1 O1']

id = data.loc[:, 'id']
entry_id = data.loc[:,'entry_id']
print(entry_id.shape)
print(id.shape)

data_save = data.loc["id","entry_id"]
data_save.to_csv("NbO.csv")