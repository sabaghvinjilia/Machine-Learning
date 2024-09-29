# Mesame Leqcia

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay

data = pd.read_csv('https://raw.githubusercontent.com/pooja2512/Adult-Census-Income/refs/heads/master/adult.csv')

# First Step ( gamotovebuli elementebis raodenoba)
print(data.isnull().sum())
# '?' -vcvlit NaN it ( radgan cxrilshi NaN is magivrad iyo '?' )
data = data.replace('?', np.nan)
print(data.isnull().sum() / data.shape[0])
# NaN -s Shlis
data.dropna(axis=0,inplace=True)

# Unique cvladebis naxva workclass svetistvis
print(data["workclass"].unique())
# Output:
# ['Private' 'State-gov' 'Federal-gov' 'Self-emp-not-inc' 'Self-emp-inc' 'Local-gov' 'Without-pay']

# Classification
def getWorkClass(text):
    if 'Private' in text:
        return 'Private'
    elif 'gov' in text:
        return 'Government'
    elif 'Self' in text:
        return 'Self'
    else:
        return text
data['workclass'] = data['workclass'].map(getWorkClass)

print(data["workclass"].unique())

# Output
# ['Private' 'Government' 'Self' 'Without-pay']

print(data["education"].unique())

# Output
# ['HS-grad' '7th-8th' 'Some-college' '10th' 'Doctorate' 'Prof-school'
#  'Bachelors' 'Masters' '11th' 'Assoc-voc' '1st-4th' '5th-6th' 'Assoc-acdm'
#  '12th' '9th' 'Preschool']

def getSchoolStatus(text):
    if 'th' in text:
        return 'School'
    else:
        return text
data['education'] = data['education'].map(getSchoolStatus)
print(data["education"].unique())

# LabelEncoder
mylabel = LabelEncoder()
for column in data.columns:
  if data[column].dtype == 'object':
    data[column] = mylabel.fit_transform(data[column])
data.head()

y = data['income']
X = data.drop('income',axis=1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state=1)


scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

reduction = PCA(n_components=2)
X_train = reduction.fit_transform(X_train)
X_test = reduction.transform(X_test)

print(np.sum(reduction.explained_variance_ratio_))

model = SVC()
model.fit(X_train,y_train)
model.score(X_test,y_test)

disp = DecisionBoundaryDisplay.from_estimator(model,
                                              X_train,
                                              response_method="predict",
                                              xlabel="Feature 1", ylabel="feature 2",
                                              alpha=0.5,
                                              cmap=plt.cm.coolwarm)

disp.ax_.scatter(X_train[:, 0], X_train[:, 1],
                 c=y_train, edgecolor="k",
                 cmap=plt.cm.coolwarm)

plt.title(f"Decision surface for tree trained on Feature 1 and Feature 2")
plt.show()

print(data.head())

# საშინაო დავალება რამოდენიე სვეტი ავიღოთ data['sveti1', 'sveti2'] რაღაც ასე ჩატიკოს ვკითხოთ