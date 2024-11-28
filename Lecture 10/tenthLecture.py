# pip install imbalanced-learn

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import _smote, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
from sklearn.model_selection import train_test_split

# Importing Algorithms
from sklearn.ensemble import StackingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

credit = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/refs/heads/master/GermanCredit.csv')
label = LabelEncoder()

print(credit["credit_risk"].value_counts())

credit["status"] = label.fit_transform(credit["status"])
print(credit["status"].unique())

credit["credit_history"] = credit["credit_history"].map(lambda x:0 if 'delay' in x else 1)
print(credit["credit_history"].value_counts())

credit["purpose"] = credit["purpose"].map(lambda x:1 if 'business' in x else 0)
print(credit["purpose"].value_counts())

credit['savings'] = label.fit_transform(credit['savings'])
print(credit["savings"].value_counts())

credit['personal_status_sex'] = credit['personal_status_sex'].map(lambda x:x.split(":")[0])
print(credit['personal_status_sex'].value_counts())

credit['personal_status_sex'] = label.fit_transform(credit['personal_status_sex'])
print(credit['personal_status_sex'].value_counts())

# მთლიანი მონაცემები გადავა ბინარულში
for column in credit.columns:
    if credit[column].dtype=='object':
        credit[column] = label.fit_transform(credit[column])

print(credit['credit_risk'].value_counts())

sampler =Pipeline(steps = [('over', SMOTE(sampling_strategy = 0.7)), ('under', RandomUnderSampler(sampling_strategy = 0.98))])

y = credit["credit_risk"]
X = credit.drop("credit_risk", axis=1)

X_new,y_new = sampler.fit_resample(X,y)

print(Counter(y_new))

X_train,X_test,y_train,y_test = train_test_split(X_new,y_new, test_size=0.2, random_state=1)

model1 = VotingClassifier(estimators=[('algo1',DecisionTreeClassifier()),('algo2',SVC(probability=True))],voting='soft')

model1.fit(X_train,y_train)
print('Model 1:', model1.score(X_test,y_test))

model2 = StackingClassifier(estimators=[('algo1',DecisionTreeClassifier()),('algo2',SVC(probability=True))],final_estimator=LogisticRegression(max_iter=70000))
model2.fit(X_train,y_train)
print('Model 2:', model2.score(X_test,y_test))

print(credit.head())