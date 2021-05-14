import numpy as np
import pandas as pd
import copy

from sklearn import svm
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def Logistic():
    modelLR = LogisticRegression()
    modelLR.fit(x_train, y_train)
    print('Logistic regression: ')
    print("Model accuracy is", modelLR.score(x_test, y_test))
    acc1 = modelLR.score(x_test, y_test)
    return acc1

def SVM():
    modelSVMlinear=svm.SVC(kernel='linear', probability=True)
    modelSVMlinear.fit(x_train,y_train)
    print('SVM with linear: ')
    print("Model accuracy is", modelSVMlinear.score(x_test, y_test))
    acc2 = modelSVMlinear.score(x_test, y_test)
    return acc2

df = pd.read_csv('./Telecom Users KTDK-1.csv')

df = df.drop(['Unnamed: 0'], axis=1)
df['Contract'] = df['Contract'].replace('Two year', 24)
df['Contract'] = df['Contract'].replace('One year', 12)
df['Contract'] = df['Contract'].replace('Month-to-month', 1)
df['PaymentMethod'] = df['PaymentMethod'].replace('Electronic check', 1)
df['PaymentMethod'] = df['PaymentMethod'].replace('Mailed check', 2)
df['PaymentMethod'] = df['PaymentMethod'].replace('Bank transfer (automatic)', 3)
df['PaymentMethod'] = df['PaymentMethod'].replace('Credit card (automatic)', 4)

df = df.replace('Yes',1)
df = df.replace('No',0)

df['gender'] = df['gender'].replace('Male', 1)
df['gender'] = df['gender'].replace('Female', 0)
df['MultipleLines'] = df['MultipleLines'].astype(int)
df['MonthlyCharges'] = df['MonthlyCharges'].astype(int)
df['TotalCharges'] = df['TotalCharges'].astype(str)
df['InternetService'] = df['InternetService'].replace('Fiber optic', 2)
df['InternetService'] = df['InternetService'].replace('DSL', 1)
df['TotalCharges'] = df['TotalCharges'].astype(float)
total_charge = df['TotalCharges']
df['TotalCharges'] = df['TotalCharges'].astype(int)

x = df.drop(['Churn'], axis=1)
y = df['Churn']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Feature scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
old = copy.deepcopy(x_train)
x_train = copy.deepcopy(old)

# Weight would contain your preferred weights for the respective features
weight = np.zeros((15, 1))

# Enter values in weight
# for i in range(0, 15):
#     weight[i] = round(float(temp[i+1]), 4)

for i in range (0, 15):
    x_train[:,i] *= weight[i]

acc1 = Logistic()
acc2 = SVM()