# NbO-Crystal-GAN

### LOG
##### 2021-2-14 Log 1
- 完成NosingFunction，其逻辑在于：任意选择batch中的一个结构作为ground并以此为基础构建generator;

##### 2021-2-14 Log 2
- 将Generator和Discriminator均转换为ConvBased网络，其效果较好;


### TODO
- [Done] 在train.py中加入noising function，基于该循环对应的结构
- [] 提高训练精度
- [ ] NAS-GAN
 
### Install and Usage

Please make sure you have python version > 3.6 and torch > 1.7

`pip3 install -r requirments.txt`

Make sure you have install OQMD database in your computer or workstation in your local aera network

Change line 8 - 10 in 0.import_OQMD.py to your MySQL database setting

Run the following command

`python 1.download_POSCAR.py`

`python 2.import_POSCAR.py`

`python 3.data_augmentation.py`

`python 4.model_training.py`

`python 5.data_post_process.py`