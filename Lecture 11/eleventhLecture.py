# Diabetes Classification
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.src.models import Sequential
from keras.src.layers import Dense
from sklearn.preprocessing import MinMaxScaler

diabetes = pd.read_csv("https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/refs/heads/master/diabetes.csv")

# Check Missing Variables
diabetes.isnull().sum()

y = diabetes['Outcome'].values
X = diabetes.drop('Outcome', axis=1).values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

y_test.astype(int)
X_test.astype(int)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()
model.add(Dense(units=30,activation='relu',input_dim=X_train.shape[1]))
model.add(Dense(units=20,activation='relu'))
model.add(Dense(units=10,activation='relu'))
model.add(Dense(units=5,activation='relu'))
model.add(Dense(units=4,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))

model.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=100,batch_size=20,validation_data=(X_test,y_test),verbose=1)

print(diabetes.corr()) # თუ კორელაცია დაბალია PCA არ გამოგვადგება

# გავზართოთ შრეების რაოდენობა
# სვეტები გავაერთიანო. მაგალითად გლუკოზა გავაერთიანოთ სისხლის წნევასთან

# optional  შედეგი გავაუმჯობესოთ