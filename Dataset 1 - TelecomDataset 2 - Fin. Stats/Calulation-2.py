import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, f_classif, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

df = pd.read_csv('./bank_final.csv')
df['Bankrupt?'].value_counts()

X = df.drop(labels=['Bankrupt?'], axis=1)
y = df['Bankrupt?']

columns = X.columns

oversample = SMOTE()
X,y = oversample.fit_resample(X,y)

scaler = StandardScaler()
X_scale = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scale, y,test_size=0.3,random_state=42)

print("Using Statistical Tests\n")

print("Mutual Information Function:")
info = mutual_info_classif(X_train, y_train)
info = pd.Series(info)
info.index = columns
print(info)
print("\n")

print("Univariate Selection using ANOVA Test:")
bestfeatures = SelectKBest(score_func=f_classif, k=30)
fit = bestfeatures.fit(X_train, y_train)
info = pd.Series(fit.scores_)
info.index = columns
print(info)
print("\n")

print("Co-relation Matrix:")
corrmat = df.corr()
top_corr_features = corrmat.index
info = abs(df.corr()['Bankrupt?'])
print(info)
print("\n")

print("Using Machine Learning Techniques\n")

print("Tree Based Classifier:")
model = ExtraTreesClassifier()
model.fit(X_train ,y_train)
info = pd.Series(model.feature_importances_)
info.index = columns
print(info)