import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)
mylabel =LabelEncoder()# this is class instance
data =pd.read_csv("https://raw.githubusercontent.com/timothypesi/Data-Sets-For-Machine-Learning-/main/penguins_cleaned.csv")
data['species'] =mylabel.fit_transform(data['species'])
data['island'] =mylabel.fit_transform(data['island'])
data['sex'] =mylabel.fit_transform(data['sex'])

print(data.head())

y = data['sex'].values
# ყველაფერი 'sex' ის გარდა
X = data.drop('sex', axis=1).values # axis = 1 ეზებს სვეტებში და axis = 0 ეწებს რიგებში (ცხრილებში 2D - data)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0) # 'test_size=0.2' ის მაგივრად შეგვიძლია დავწეროთ 'stratify=y'

# ვიყენებთ მანძილის ალგორითმს

model.fit(X_train,y_train)
print(model.score(X_test,y_test))

# output - 0.83 ნიშნავს რომ 100 მონაცემიდან 83%-ს გამოიცნობს დანარჩენი არის Error & Accuracy