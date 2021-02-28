# NbO-Crystal-GAN

### LOG
##### 2021-2-14 Log 1
- 完成NosingFunction，其逻辑在于：任意选择batch中的一个结构作为ground并以此为基础构建generator;

##### 2021-2-14 Log 2
- 将Generator和Discriminator均转换为ConvBased网络，其效果较好;

##### 2021-2-16 Log 3
- 将Generator改为ConvTransp网络，效果较好，产生了有效结构

##### 2021-2-16 Log 4
- 新增VASP检测模块


### TODO
- [x] Add nosing function in 4.model_training.py, based on structure used in this epoch
- [x] Further improve accuracy
- [ ] NAS-GAN
- [ ] Applicable to ternary systems
- [x] Add "Count Element" code
 
### Install and Usage

##### Install necessary prerequisite

Please make sure you have python version > 3.6 and torch > 1.7

`pip3 install -r requirments.txt`

##### Setting

Make sure you have install OQMD database in your computer or workstation in your local aera network. The install step may see in [my ZhiHu page](https://zhuanlan.zhihu.com/p/345722370).

Change line 8 - 10 in 0.import_OQMD.py to your MySQL database setting. For local installation, `[host address] `shoud be `localhost` or `127.0.0.1`.

```python
con = pymysql.connect(host = "[host address]", 
           user = "[Your user name]", password = '[Your Password]', 
           db = "qmdb", charset='utf8')
```

##### Run

Run the following command in working direction

`python 1.download_POSCAR.py`

`python 2.import_POSCAR.py`

`python 3.data_augmentation.py`

`python 4.model_training.py`

`python 5.data_post_process.py`

Before the following command, make sure you have properly installed `VASP` and `intel parallel studio` in your local computer or work station. VASP 5.4 and above is recommanded.

`python 6.vasp_test.py`

`python 7.result_eval.py`

After that, you might have to manually adjust perdicted structures and debug. 
