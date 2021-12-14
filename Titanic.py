#!/usr/bin/env python
# coding: utf-8

# In[1]:


#导入numpy、pandas数据包
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

#导入数据集
#导入训练数据集
train=pd.read_csv(r'C:\Users\Lenovo\Desktop\机器学习\课设\train.csv')
#导入测试数据集
test=pd.read_csv(r'C:\Users\Lenovo\Desktop\机器学习\课设\test.csv')


#合并数据集，方便对数据进行清洗
allData=train.append(test,ignore_index=True)


#数据类型缺失值处理

#Age缺失值处理，缺失1309-1046=263，缺失率263/1309=20%，用平均值代替
allData['Age']=allData['Age'].fillna(allData['Age'].mean())

#Fare（船票价格）缺失值处理，缺失1309-1308=1，用平均值代替
allData['Fare']=allData['Fare'].fillna(allData['Fare'].mean())


#字符串列数据缺失值处理

#Embarked（登船港口）缺失值处理，缺失1条，缺失较少，该数据为字符串数据，为分类变量，看下最常见的类别，用其填充
allData['Embarked'].value_counts()
allData['Embarked']=allData['Embarked'].fillna('S')

#乘客性别（Sex）数据处理
#将性别的值映射为数值，男male对应数值1，女female对应数值0
sex_map={'male':1,'female':0}
#map函数，对series每个数据应用自定义的函数计算
allData['Sex']=allData['Sex'].map(sex_map)

#登船港口（Embarked）数据处理
embarked_map={'S':1,'C':2,'Q':3}
#map函数，对series每个数据应用自定义的函数计算
allData['Embarked']=allData['Embarked'].map(embarked_map)

allData['FamilySize'] = allData['SibSp'] + allData['Parch'] + 1

# allData.head()

# # Sex
# sns.countplot('Sex', hue='Survived', data=train)
# plt.show()

corrDf=allData.corr()
corrDf['Survived'].sort_values(ascending=False)

# 获得即将用于测试的数据
data_X=pd.concat([allData['Pclass'],allData['FamilySize'],allData['Parch'],allData['Age'],allData['Fare'],allData['Embarked'],allData['Sex']],axis=1)
#原始数据集：特征
source_X=data_X.loc[0:891-1,:]
#原始数据集：标签
source_Y=allData.loc[0:891-1,'Survived']

#预测数据集：特征
pred_X=data_X.loc[891:,:] #从891行开始到最后一行作为预测数据集

from sklearn.model_selection  import train_test_split

#拆分建立模型用的训练数据集和测试数据集
train_X,test_X,train_Y,test_Y=train_test_split(source_X,source_Y,train_size=.8)

#导入算法
from sklearn.linear_model import LogisticRegression
#创建模型：逻辑回归（logic regression）
model=LogisticRegression(max_iter=1000)
#训练模型
model.fit(train_X,train_Y)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)

#得到正确率
model.score(test_X,test_Y)

#使用机器学习模型，对预测数据集中的生存情况进行预测
pred_Y=model.predict(pred_X)

pred_Y=pred_Y.astype(int)

#乘客id
passenger_id = allData.loc[891:,'PassengerId']
#数据框：乘客id，预测生存情况的值
predDf = pd.DataFrame( 
    { 'PassengerId': passenger_id , 
     'Survived': pred_Y } )

#保存结果
predDf.to_csv( r'C:\Users\Lenovo\Desktop\Tianic.csv' , index = False )


# In[ ]:




