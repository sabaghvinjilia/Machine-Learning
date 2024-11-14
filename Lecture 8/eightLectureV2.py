# Imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import plot_tree

data = pd.read_csv('https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/refs/heads/master/diabetes.csv')

print(data.corr())

y = data['Outcome']
X = data.drop('Outcome', axis=1)
Feature_names = X.columns

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

model = RandomForestClassifier(n_estimators=5,criterion='entropy',max_depth=3)
model.fit(X_train,y_train)
model.score(X_test,y_test)

print(model.estimators_)
print(model.estimators_[1])

# plot_tree(model.estimators_[0],max_depth=3,feature_names=Feature_names,class_names=['Diabetes','Health'],fontsize=4)
plt.show()

hybrid =Pipeline(steps =[("scaler",MinMaxScaler()),
                         ("pca",PCA()),
                         ("model",RandomForestClassifier())])

hybrid.fit(X_train,y_train)
print(hybrid.score(X_test,y_test))

parameters = {
    "scaler__feature_range": [(0, 1), (1, 2)],
    "pca__n_components": [3, 4, 5],
    "model__n_estimators": [20, 40, 60, 80]
}

giant = GridSearchCV(estimator=hybrid, param_grid=parameters,scoring="accuracy",n_jobs=1,cv=3,verbose=4)
giant.fit(X_train,y_train)
print(giant.score(X_test,y_test))
# საუკეთესო ვარიანტი
print(giant.best_params_)