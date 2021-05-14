import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import ExtraTreesClassifier

df = pd.read_csv('./Telecom Users KTDK-1.csv')

df = df.drop(['Unnamed: 0'], axis=1)
df['Contract'] = df['Contract'].replace('Two year', 24)
df['Contract'] = df['Contract'].replace('One year', 12)
df['Contract'] = df['Contract'].replace('Month-to-month', 1)
df['PaymentMethod'] = df['PaymentMethod'].replace('Electronic check', 1)
df['PaymentMethod'] = df['PaymentMethod'].replace('Mailed check', 2)
df['PaymentMethod'] = df['PaymentMethod'].replace('Bank transfer (automatic)', 3)
df['PaymentMethod'] = df['PaymentMethod'].replace('Credit card (automatic)', 4)

df = df.replace('Yes', 1)
df = df.replace('No', 0)

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

columns = df.columns.drop('Churn')

x = df.drop(['Churn'], axis=1)
y = df['Churn']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Feature scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

print("Using Statistical Tests\n")

print("Mutual Information Function:")
info = mutual_info_classif(x_train, y_train)
info = pd.Series(info)
info.index = columns
info = info.sort_values(ascending=False)
print(info)
print("\n")

print("Univariate Selection using ANOVA Test:")
bestfeatures = SelectKBest(score_func=f_classif, k=10)
fit = bestfeatures.fit(x_train, y_train)
info = pd.Series(fit.scores_)
info.index = columns
info = info.sort_values(ascending=False)
print(info)
print("\n")

print("Co-relation Matrix:")
corrmat = df.corr()
top_corr_features = corrmat.index
info = abs(df.corr()['Churn'])
info = info.sort_values(ascending=False)
print(info)
print("\n")

print("Using Machine Learning Techniques\n")

print("Tree Based Classifier:")
model = ExtraTreesClassifier()
model.fit(x_train ,y_train)
info = pd.Series(model.feature_importances_)
info.index = columns
info = info.sort_values(ascending=False)
print(info)