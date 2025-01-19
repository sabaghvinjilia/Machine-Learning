import pandas as pd
from sklearn.model_selection import train_test_split
from keras.src.models import Sequential
from keras.src.layers import Dense
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/refs/heads/master/banknote_authentication.csv', header=None)
data.columns = ['variance','skewness','curtosis','entropy','class']

y = data['class'].values
X = data.drop('class',axis=1).values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()
model.add(Dense(units=30,activation='relu',input_dim=X_train.shape[1]))
model.add(Dense(units=30,activation='relu'))
model.add(Dense(units=30,activation='relu'))
model.add(Dense(units=30,activation='relu'))
model.add(Dense(units=30,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))

model.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=30,batch_size=20,validation_data=(X_test,y_test),verbose=1)

print(data.head())