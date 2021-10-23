#!/usr/bin/env python
# coding: utf-8

# In[3]:


import csv
import json


# In[1]:


# 파일 리스트 불러오기
import pandas as pd
import os
path = 'C:/Users/parkjj/Stock_csv/'
list_csv = (os.listdir("C:/Users/parkjj/Stock_csv"))
print(list_csv)


# In[2]:


# 각 1_ , 2_ ~ 로 나눠 정렬하기
k = 1
list_1 = []
list_2 = []
list_3 = []
for i in range(1,10):
    for j in range(0,3):
        print(list_csv[k +j])
        if j == 0 :
            list_1.append(path+list_csv[k+j])
        if j == 1 :
            list_2.append(path+list_csv[k+j])
        if j == 2 :
            list_3.append(path+list_csv[k+j])
    k += 3
    print('--------------')
print(list_1)
print('--------------')
print(list_2)
print('--------------')
print(list_3)


# In[83]:


# 각 주식별로 그래프 그리기
import matplotlib.pyplot as plt
# 필요한 모듈 import 하기 
import plotly
import plotly.graph_objects as go
import plotly.express as px

# %matplotlib inline 은 jupyter notebook 사용자용 - jupyter notebook 내에 그래프가 그려지게 한다.
get_ipython().run_line_magic('matplotlib', 'inline')
for i in range(0,9):
    df1 = pd.read_csv(list_1[i])
    df2 = pd.read_csv(list_2[i])
    df3 = pd.read_csv(list_3[i])
    
    plt.figure(figsize=(10,4))   
    plt.plot(df1['Date'], df1['Close'],label = list_csv[i * 3+1][2:-4]) 
    plt.plot(df2['Date'],df2['Close'],label = list_csv[i * 3 + 2][2:-4] )
    plt.plot(df3['Date'], df3['Close'],label = list_csv[i * 3 + 3][2:-4] )
    plt.legend(loc = 'upper right')
    plt.xlabel('Date')
    plt.ylabel('Close')
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    plt.savefig('company' +str(i)+ ".png")
    plt.show()
    df1_c = df1.columns
    df2_c = df2.columns
    df3_c = df3.columns
    df1.rename(columns = {'Close' : list_csv[i * 3+1][2:-4]}, inplace = True) 
    df2.rename(columns = {'Close' : list_csv[i * 3+2][2:-4]}, inplace = True) 
    df3.rename(columns = {'Close' : list_csv[i * 3+3][2:-4]}, inplace = True) 

    if i == 0 :
        all_data = pd.concat((df1['Date'], df1[list_csv[i * 3+1][2:-4]], df2[list_csv[i * 3+2][2:-4]],df3[list_csv[i * 3+3][2:-4]]),axis = 1)
    elif i != 0:
        all_data = pd.concat((all_data, df1[list_csv[i * 3+1][2:-4]], df2[list_csv[i * 3+2][2:-4]],df3[list_csv[i * 3+3][2:-4]]),axis = 1)


# In[4]:


print(all_data)


# In[8]:


# 코로나 데이터 불러오기
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

# pd.read_csv를 통해 Dataframe의 형태로 csv 파일을 읽어옴
corona_all = pd.read_csv("./Seoul_Covid.csv", encoding='cp949')

corona_all.head()

corona_all.info()

# drop 함수를 사용하여 불필요한 정보의 column 데이터를 삭제
corona_del_column = corona_all.drop(columns = ['환자번호', '국적', '환자정보', '지역', '여행력', '접촉력','조치사항', '상태', '이동경로', '등록일', '수정일', '노출여부'])

# dropna()를 이용하여 결측값 있는 행 제거
corona_del_column = corona_del_column.dropna()

# rename()을 이용하여 한글로 된 컬럼명을 영어로 바꿈
corona_del_column = corona_del_column.rename(columns = {'연번':'number', '확진일':'date'})

corona_del_column.info()

days = corona_del_column['date']

# 확인일 데이터를 년, 월, 일별 데이터로 나누기
year = []
month = []
day = []

for data in corona_del_column['date']:
    # split 함수를 사용하여 년, 월, 일을 나누어 리스트에 저장
    year.append(data.split('-')[0])
    month.append(data.split('-')[1])
    day.append(data.split('-')[2])

corona_del_column['year'] = year
corona_del_column['month'] = month
corona_del_column['day'] = day
corona_del_column['year'].astype('int64')
corona_del_column['month'].astype('int64')
corona_del_column['day'].astype('int64')

corona_del_column.head()

# 월별 확진자 데이터 1월->01월
year_2020 = corona_del_column['year'] == '2020'
year_2021 = corona_del_column['year'] == '2021'

month_arr = []
for i in range(1, 13):
    i = str(i)
    if(len(i)%2 == 1):
        i = "0"+str(i)
    month_arr.append(i)
print(month_arr)

corona_month_2020 = []
corona_month_2021 = []

for j in month_arr:
    month_2020 = corona_del_column['month'] == j
    corona_month_2020.append(len(corona_del_column[year_2020 & month_2020]))
    month_2021 = corona_del_column['month'] == j
    corona_month_2021.append(len(corona_del_column[year_2021 & month_2021]))
print(corona_month_2021)


# In[9]:


list_day = np.array(days.tolist())
print(list_day)


# In[85]:


list_pro = []
drop_num = []
a = 0
pre_stock = all_data
stock_col = all_data['Date']
covid = pd.read_csv('seoul_covid_pro.csv')
covid_col = covid['Date']
print(len(stock_col))
print(len(covid_col))
print(len(covid['Num']))
for i in range(len(stock_col)):
    a = 0
    for j in range(len(covid_col)):
        if stock_col[i] == covid_col[j]:
            list_pro.append(covid['Num'][j])
            drop_num.append(covid_col[j])
            a = 1
    if a == 0 :
            list_pro.append(0)
            drop_num.append(covid_col[j])
print(len(list_pro))
print(len(drop_num))

print(len(list_pro))


# In[86]:


co_num = pd.DataFrame(list_pro)
co_date = pd.DataFrame(drop_num)
#print(co_num)
#print(co_date)
print(co_num.columns[0])
co_num.columns = ['Co_num']
print(co_num.columns[0])
su_data = pd.concat((all_data, co_num),axis = 1)
print(su_data)


# In[87]:


corr = su_data.corr(method='pearson')
print(corr)


# In[88]:


sort_corr = corr.sort_values(by = ['Co_num'], ascending=False)
print(sort_corr['Co_num'])
print(sort_corr['Co_num'].index.values)


# In[15]:


corr['Co_num'].to_csv('corr.csv', header = True)


# In[82]:


from matplotlib import pyplot

import numpy as np

X_axis = np.array(sort_corr['Co_num'].tolist())
Y_axis = np.array(sort_corr['Co_num'].index.values.tolist())
print(X_axis)
print(Y_axis)

pyplot.rcParams['font.family'] = 'Malgun Gothic'
pyplot.rcParams['font.size'] = 12
pyplot.rcParams['figure.figsize'] = (12, 8)
pyplot.rcParams['axes.unicode_minus'] = False
pyplot.barh(list(reversed(Y_axis)), list(reversed(X_axis)), label = 'Pearson Correlation Coefficient')
pyplot.legend()
pyplot.ylabel('Name of Stocks')
pyplot.xlabel('Pearson Correlation Coefficient')
#pyplot.xlim(-1, 1)
pyplot.title('Pearson Correlation Coefficient of Stocks')
pyplot.yticks(Y_axis)
pyplot.grid()
pyplot.tight_layout()
pyplot.savefig('shap.png')
pyplot.show()
pyplot.close()


# In[18]:


su_data.to_csv('last_data.csv', header = True)


# In[57]:


import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
dataset = pd.read_csv('last_data.csv')
print(dataset)
data_col = dataset.columns
print(data_col)
#dataset = shuffle(dataset)
#print(dataset[9:])
dataset = dataset.values
dataset = dataset[:,2:].astype(float)
print(dataset.shape)
x_train = dataset[ 9 : , [27]]
y_train = dataset[9: ,0:27]
x_val = dataset[:9 , [27]]
y_val = dataset[:9 , 0:27]


# In[89]:


import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from keras import models, optimizers, layers, regularizers
from keras.layers.normalization import layer_normalization
from keras.callbacks import EarlyStopping
from keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow as tf 
from keras import models
from keras import layers
from pandas import DataFrame as df
for i in range(27):

  model =  models.Sequential()
  model.add(layers.Dense(1, activation='elu', kernel_regularizer=regularizers.l1_l2(0.0001),input_shape=(1,)))
  model.add(layers.Dense(3, activation='elu'))
  model.add(layers.Dense(2, activation='elu'))
  model.add(layers.Dense(1, activation='elu'))
  model.add(layers.Dense(4, activation='elu'))
  model.add(layers.Dense(1))
  adam = tf.optimizers.Adam(lr=0.04, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.002)
  model.compile(optimizer=adam, loss='mse' , metrics=['mae'])

  model.compile(optimizer = 'adam' , loss = 'mse', metrics=['mae'])

  early_stopping = EarlyStopping(patience=100)

  history = model.fit(x_train, y_train[ :, [i]] ,epochs=1000, batch_size=50, verbose=0 ,
                    validation_data = (x_val, y_val[:, [i]])
                    ,callbacks = [early_stopping])
  predictions = model.predict(x_val)
  if i == 0:
    b = predictions
    c = y_val[:, [0]]
  else :
    b =     (np.concatenate((b, predictions), axis=1))
    c =     (np.concatenate((c, y_val[ : , [i]]), axis=1))
  print(b)
  print(c)
  plt.clf()

  plt.plot(predictions, y_val[:,[i]], 'ro', label = 'prediction')
  plt.title(data_col[i+2])
  plt.xlabel('prediction')
  plt.ylabel('answer')
  plt.legend()
  plt.savefig(str(i) +'png')
  plt.show()


# In[ ]:


with open('test.csv', 'w',newline='') as f: 
      
    # using csv.writer method from CSV package 
    write = csv.writer(f) 
      
    write.writerows(b)
    
with open('ans.csv', 'w',newline='') as f: 
      
    # using csv.writer method from CSV package 
    write = csv.writer(f) 
      
    write.writerows(c)
    
aa = 0
aaa = []
import math
for i in range(27):
  for j in range(9):
    a = math.sqrt((b[j][i] - c[j][i]) * (b[j][i] - c[j][i]))
    aa += a
  try:
      aaa.append(int(aa))
  except ValueError:
      aaa.append(0)
  aa = 0

print((aaa))  

