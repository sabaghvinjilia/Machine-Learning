import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv('data.csv')
mylabel = LabelEncoder()


#   1
data["statezip"]=data["statezip"].map(lambda x: x[0])

def first_letter(address):
    return address.split()[0]

data["state"] = data["statezip"].map(first_letter)
print(data['state'])

data['street'] = mylabel.fit_transform(data['street'])
data['city'] = mylabel.fit_transform(data['city'])
data['statezip'] = mylabel.fit_transform(data['statezip'])
data['country'] = mylabel.fit_transform(data['country'])
data['state'] = mylabel.fit_transform(data['state'])

#   2
# Svetebis Sheqmna
# Floors + Waterfront + View + Condition = FWVC
data['FWVC'] = data['floors'] + data['waterfront'] + data['view'] + data['condition']
# Svetebis Gamotana
print(data['FWVC'])
# Svetebis Washla
data.drop(['floors', 'waterfront', 'view', 'condition'], axis=1, inplace=True)

print(data.head())

#   3

y = data['price'].values
X = data.drop('price', axis=1).values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.35,random_state=0)

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print(y_pred)
score = model.score(X_test,y_test)
print(score)