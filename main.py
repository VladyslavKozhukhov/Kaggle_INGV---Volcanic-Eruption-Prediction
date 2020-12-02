# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import scipy
import scipy.signal
import datetime
import matplotlib
import matplotlib.pyplot as plt
import tqdm
import random
import os
import glob
from pathlib import Path

#from google.colab import drive
#drive.mount('/content/gdrive')

#Defines
#/drive/MyDrive/Kaggle/predict-volcanic-eruptions-ingv-oe
train_csv_path = "./gdrive/MyDrive/Kaggle/predict-volcanic-eruptions-ingv-oe/train.csv"
sample_csv_path = "./gdrive/MyDrive/Kaggle/predict-volcanic-eruptions-ingv-oe/sample_submission.csv"
single_segment = "./gdrive/MyDrive/Kaggle/predict-volcanic-eruptions-ingv-oe/train/1003520023.csv"
train_data_path = "./gdrive/MyDrive/Kaggle/predict-volcanic-eruptions-ingv-oe/train/"
test_data_path = "./gdrive/MyDrive/Kaggle/predict-volcanic-eruptions-ingv-oe/test/"
sample_size = 10

train = pd.read_csv(train_csv_path)
sample_submission = pd.read_csv(sample_csv_path)

# Convert 'time_to_eruption'to hours:minutes:seconds (Just for reference)
#train['days_hours_minutes_seconds'] = (train['time_to_eruption']
             #     .apply(lambda x:datetime.timedelta(seconds = x/100))) 100???? why

train

train['time_to_eruption'].describe()

print('Median:', train['time_to_eruption'].median())
print('Skew:', train['time_to_eruption'].skew()) #symetric?
print('Std:', train['time_to_eruption'].std())
print('Kurtosis:', train['time_to_eruption'].kurtosis()) #heavy-tailed?
print('Mean:', train['time_to_eruption'].mean())

sample_submission

# From DF to Dic for O(1) access
train_dic = dict(zip(train.segment_id, train.time_to_eruption))
train_dic

"""Look on single segment

"""

segment = pd.read_csv(single_segment,dtype ="int16")
segment.describe()

segment.fillna(0).plot(subplots=True, figsize=(25, 10))
plt.tight_layout()
plt.show()

"""Merge data for train/test"""

train_segmnets = glob.glob(os.path.join(train_data_path,"*.csv"))
test_segmnets = glob.glob(os.path.join(test_data_path,"*.csv"))

train_segmnets

#lst_of_segments_data = []
#lst_of_segments_number = []
#read each segment and save inside li
#for filename in train_segmnets:
#    df = pd.read_csv(filename, index_col=None, header=0)
#    lst_of_segments_data.append(df)
#    lst_of_segments_number.append(int((filename.split('/')[-1])[:-4]))
#lst_of_segments_data[1]
#frame = pd.concat(li, axis=0, ignore_index=True)
#frame

"""Random sample data and split for training and test
for internal using
"""

#sampling with replacement
sampling_segmentID = random.choices(train_segmnets, k=sample_size )
sampling_segmentID

sampling_segmentID = [int(x.split('/')[-1][:-4]) for x in sampling_segmentID]
sampling_segmentID

#Calculate x_train,y_train,x_test,y_test

y_train =[]
x_train = []
x_test = []
y_test = []
counter = 0
for seg in sampling_segmentID:## 3 test 7 train
  if counter <3:
    y_test.append(train_dic[seg])
    tmpDf = pd.read_csv(train_data_path+str(seg)+".csv", index_col=None, header=0)
    mean_per_collumn = tmpDf.mean(axis = 0).fillna(tmpDf.mean(axis = 0).mean())
    x_test.append(np.array(a))
    counter = counter +1
  else:
    y_train.append(train_dic[seg])
    tmpDf = pd.read_csv(train_data_path+str(seg)+".csv", index_col=None, header=0)
    mean_per_collumn = tmpDf.mean(axis = 0).fillna(tmpDf.mean(axis = 0).mean())
    x_train.append(np.array(a))

"""Logistic regression"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly_model = make_pipeline(PolynomialFeatures(10),
                           LinearRegression())
poly_model.fit(x_train, y_train)
yfit = poly_model.predict(x_test)
mae = mean_absolute_error( y_test,yfit)
mae
#3843230 - 1 place
#11598807 - Our result

