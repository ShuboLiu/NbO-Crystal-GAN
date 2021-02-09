import pandas as pd 
import numpy as np 
import xlrd

raw_data = np.load(r"fake_imgs_gen.npy")
print(raw_data.shape)
print(raw_data)