import numpy as np
import pandas as pd
import copy

from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

def Logistic():
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    log_reg.score(X_test, y_test)
    y_pred = log_reg.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    acc1 = accuracy_score(y_test, y_pred)
    return acc1

def SVM():
    model = SVC()
    model.fit(X_train, y_train)
    model.score(X_test, y_test)
    svc_predict = model.predict(X_test)
    print(accuracy_score(y_test, svc_predict))
    acc2 = accuracy_score(y_test, svc_predict)
    return acc2

df = pd.read_csv('./Fin. Stats KTDK-2.csv')
df['Bankrupt?'].value_counts()

X = df.drop(labels=['Bankrupt?'], axis=1)
y = df['Bankrupt?']

oversample = SMOTE()
X,y = oversample.fit_resample(X,y)

scaler = StandardScaler()
X_scale = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scale, y,test_size=0.3,random_state=42)
old = copy.deepcopy(X_train)
X_train = copy.deepcopy(old)

# Weight would contain your preferred weights for the respective features
weight = np.zeros((30, 1))

# Enter values in weight
# for i in range(0, 30):
#     weight[i] = round(float(temp[i+1]), 4)

for i in range (0, 15):
    X_train[:,i] *= weight[i]

acc1 = Logistic()
acc2 = SVM()