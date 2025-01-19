from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from keras.src.models import Sequential
from keras.src.layers import Dense
from sklearn.model_selection import train_test_split

# Importing Algorithms
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


X, y = make_classification(n_samples=100, n_features=23,random_state=26)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X= scaler.transform(X)

model = Sequential()
model.add(Dense(units=512,activation='relu',input_dim=X.shape[1]))
model.add(Dense(units=256,activation='relu'))
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=64,activation='relu'))
model.add(Dense(units=32,activation='relu'))
model.add(Dense(units=16,activation='relu'))
model.add(Dense(units=8,activation='relu'))
model.add(Dense(units=4,activation='relu'))
model.add(Dense(units=2,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))

model.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=100,batch_size=20,validation_data=(X_test,y_test),verbose=1)

model1 = VotingClassifier(estimators=[('algorithm1',DecisionTreeClassifier()),('algorithm2',SVC(probability=True))],voting='soft')
model1.fit(X_train,y_train)
print('Model 1:', model1.score(X_test,y_test))

model2 = VotingClassifier(estimators=[('algorithm1',RandomForestClassifier()),('algorithm2',SVC(probability=True))])
model2.fit(X_train,y_train)
print('Model 2:', model2.score(X_test,y_test))

model3 = VotingClassifier(estimators=[('algorithm1',KNeighborsClassifier()),('algorithm2',SVC(probability=True))],voting='soft')
model3.fit(X_train,y_train)
print('Model 3:', model3.score(X_test,y_test))