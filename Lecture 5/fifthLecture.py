# regression
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import numpy as np

house = pd.read_csv("https://raw.githubusercontent.com/ankita1112/House-Prices-Advanced-Regression/refs/heads/master/train.csv")

myLabel = LabelEncoder()
model = LinearRegression()

house.drop('Id',axis=1,inplace=True)

house['LotFrontage'] =house['LotFrontage'].fillna(house['LotFrontage'].median())
house['MasVnrArea']=house['MasVnrArea'].fillna(house['MasVnrArea'].median())

# გამოტოვბული ელემენტი
print(house.isnull().sum)

# გამოტოვებული ელემენტების შევსება
# boxplot ით ვამოწმებთ. თუ ძალიან დიდია დაშორება მაშინ მედიანათი ვავსებთ და თუ პატარაა მაშინ საშუალო არითმეტიკულით
# ამ შემთხვევაში მედიანით შევავსეთ
house['LotFrontage'] =house['LotFrontage'].fillna(house['LotFrontage'].median())

house.boxplot()

# რომელია გამოტოვებული სვეტი და რამდენი
selected_house = pd.DataFrame({"Attributes":house.isnull().sum().index,"total_missed":house.isnull().sum().values})
selected_house[selected_house['total_missed']>0]

# MAPPING

house['Alley']=house['Alley'].map({"Grvl":0,"Pave":1})
house['Alley'] =house['Alley'].fillna(0.5)
# house['Alley'].value_counts(normalize=True)

house['BsmtQual']=house['BsmtQual'].map({"Gd":0,"Ex":1,"Fa":2,"TA":3})
house['BsmtQual']=house['BsmtQual'].fillna(0.25)    #   0.25 ით იმიტმომ ვავსებთ რომ 4 ელემენტია და 100/4=25 და 0.25-ით მაგიტომ ვავსებთ
print(house['BsmtQual'].value_counts())

house['MasVnrType']=house['MasVnrType'].map({"BrkFace":0,"BrkCmn":0,"Stone":1})
house['MasVnrType']=house['MasVnrType'].fillna(0.5) #   0.5 ით იმიტომ ვაკეთებთ, რომ სულ გვაქ 2 ელემენტი და მონაცემეიბ რომ არ დაზიანდეს საშუალოთი ვავსებთ
print(house['MasVnrType'].value_counts(normalize=True))

house['BsmtCond']=house['BsmtCond'].map({"Gd":0,"Fa":1,"Po":2,"TA":3})
house['BsmtCond']=house['BsmtCond'].fillna(0.25)
print(house['BsmtCond'].value_counts())

for  column in house.columns:
  if house[column].dtype == 'object':
    house[column] = house[column].fillna(house[column].value_counts().index[0])
  else:
    house[column] = house[column].fillna(house[column].mean())

def get_zone(text):
  if text in ['RL','RH','RM']:
    return 0
  else:
    return  1

house['MSZoning']=house['MSZoning'].map(get_zone)
house['MSZoning'].value_counts()

for column in house.columns:
    if house[column].dtype == 'object':
        house[column] = myLabel.fit_transform(house[column])

# ორი სვეტის გაერთიანება და წაშლა ( რაც შევკრიბეთ ის წავშალეთ ( განზომილება შევამცირეთ) )

house['Quality'] =( house['PoolQC']+house['Fence'] )/2
house.drop([ 'PoolQC','Fence'],axis=1,inplace=True )

y = house["SalePrice"].values
X = house.drop("SalePrice", axis=1).values

# მონაცემების გაყოფა სატრენინგოდ და სატესტოდ
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=1)

model.fit(X_train,y_train)
model.score(X_test,y_test)

pca = PCA(n_components=51)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

model.fit(X_train,y_train)
model.score(X_test,y_test)

#  რამდენი ინფორმაცია დავკარგეთ
print(np.sum(pca.explained_variance_ratio_))

print(house.head())