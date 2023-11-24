import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn. model_selection import GridSearchCV,cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix,classification_report
import  joblib as jb
pd.set_option('display.max_columns',12)
pd.set_option('display.width',300)

df=pd.read_csv(r'C:\Users\DELL\Desktop\heart.csv')
print(df)
print(df.info())
print(df.shape)
print(df.isnull().sum())
print(df.columns)
print(df.describe())
encode=LabelEncoder()
df['Sex']=encode.fit_transform(df['Sex'])
df['ChestPainType']=encode.fit_transform(df['ChestPainType'])
df['RestingECG']=encode.fit_transform(df['RestingECG'])
df['ExerciseAngina']=encode.fit_transform(df['ExerciseAngina'])
df['ST_Slope']=encode.fit_transform(df['ST_Slope'])
dis=df[df['HeartDisease']==0]
non_dis=df[df['HeartDisease']==1]
print('the number of people with heart disease:',dis['HeartDisease'].count())
print('the number of people without heart disease:',non_dis['HeartDisease'].count())
print(df)
print(df.describe())
print(df.groupby(['HeartDisease']).mean())
x=df.drop(columns='HeartDisease' )
y=df['HeartDisease']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scalar=StandardScaler()
normal_x=scalar.fit_transform(x_train)
x_training=pd.DataFrame(normal_x,columns=x.columns)
log_model=LogisticRegression()
log_model.fit(x_training,y_train)
log_pred=log_model.predict(x_training)
log_metric=confusion_matrix(y_train,log_pred)
print(log_metric)
report_log=classification_report(y_train,log_pred)
print(report_log)


ran_modeL=RandomForestClassifier()
ran_modeL.fit(x_training,y_train)
ran_pred=ran_modeL.predict(x_training)
ran_metric=confusion_matrix(y_train,ran_pred)
print(ran_metric)
report_ran=classification_report(y_train,ran_pred)
print(report_ran)

des_model=DecisionTreeClassifier()
des_model.fit(x_training,y_train)
des_pred=des_model.predict(x_training)
des_metric=confusion_matrix(y_train,des_pred)
print(des_metric)
report_des=classification_report(y_train,des_pred)
print(report_des)


knn_model=KNeighborsClassifier()
knn_model.fit(x_training,y_train)
knn_pred=knn_model.predict(x_training)
knn_metric=confusion_matrix(y_train,knn_pred)
print(knn_metric)
report_knn=classification_report(y_train,knn_pred)
print(report_knn)

cv_log=cross_val_score(log_model,x_training,y_train,cv=5,scoring='accuracy')
print('the cv log',cv_log.mean())
cv_ran=cross_val_score(ran_modeL,x_training,y_train,cv=5,scoring='accuracy')
print('the cv ran',cv_ran.mean())

cv_des=cross_val_score(des_model,x_training,y_train,cv=5,scoring='accuracy')
print('the cv des',cv_des.mean())

cv_knn=cross_val_score(knn_model,x_training,y_train,cv=5,scoring='accuracy')
print('the cv knn',cv_knn.mean())

jb.dump(ran_modeL,"model.py")
print('the columns of the x variables :', x.columns)
print(' the descrptions variables ',df.describe())