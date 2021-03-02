import pymysql
import pandas as pd
import numpy as np
import os
import xlrd

from pymysql import connect
from sshtunnel import SSHTunnelForwarder

# 指定SSH远程跳转
server = SSHTunnelForwarder(ssh_address_or_host=('10.69.21.155', 10025),  # 指定SSH中间登录地址和端口号
                            ssh_username='liusb',  # 指定地址B的SSH登录用户名
                            ssh_password='123456',  # 指定地址B的SSH登录密码
                            local_bind_address=('127.0.0.1', 3306),  # 绑定本地地址A（默认127.0.0.1）及与B相通的端口（根据网络策略配置，若端口全放，则此行无需配置，使用默认即可）
                            remote_bind_address=('localhost', 3306)  # 指定最终目标C地址，端口号为mysql默认端口号3306
                            )

server.start()
# 打印本地端口，以检查是否配置正确
print(server.local_bind_port)

# 设置mysql连接参数，地址与端口均必须设置为本地地址与端口
# 用户名和密码以及数据库名根据自己的数据库进行配置
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