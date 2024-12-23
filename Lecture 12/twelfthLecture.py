import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.src.models import Sequential
from keras.src.layers import Dense, Dropout, BatchNormalization

from thirdLecture import X_train


def travel_info(text):
    if "Frequently" in text:
        return 2
    elif "Rarely" in text:
        return 1
    else:
        return 0

def education_status(text):
    if "Other" in text or "Human" in text:
        return 0
    else:
        return 1

def get_job(status):
    if "Sales" in status:
        return 1
    elif "Human" in status:
        return 0
    else:
        return 0

data = pd.read_csv('https://raw.githubusercontent.com/nelson-wu/employee-attrition-ml/refs/heads/master/WA_Fn-UseC_-HR-Employee-Attrition.csv')

myLabel = LabelEncoder()
scaler = MinMaxScaler()
model = Sequential()
model.add(Dense(units=90,activation='relu',input_dim =X_train.shape[1]))
model.add(Dropout(rate=0.5))
model.add(Dense(units=60,activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(units=50,activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(units=40,activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(units=20,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=1,activation='sigmoid'))
model.compile(optimizer='sgd',loss='binary_crossentropy', metrics=['accuracy'])

#First Stage is to check a missing values
print(data.isnull().sum())

#Second. Check Types of each column
print(data.info())

#Select only objects
print(data.select_dtypes(include='object'))
print(data["Attrition"].value_counts())

#Map
data["Attrition"] = data["Attrition"].map({"No":0, "Yes":1})

#BusinessTravel
print(data["BusinessTravel"].value_counts())
data["BusinessTravel"] = data["BusinessTravel"].map(travel_info)

#Department
print(data["Department"].value_counts())
data["Department"] = data["Department"].map(lambda x: 0 if "Human" in x else 1)
print(data["Department"].value_counts())

#EducationField
print(data["EducationField"].value_counts())
data["EducationField"] = data["EducationField"].map(education_status)
print(data["EducationField"].value_counts())

#Gender
print(data["Gender"].value_counts())
data["Gender"] = data["Gender"].map({"Female":0, "Male":1})
print(data["Gender"].value_counts())

#JobRole
print(data["JobRole"].value_counts())
data["JobRole"] = data["JobRole"].map(get_job)
print(data["JobRole"].value_counts())

#LabelEncoder
data["MaritalStatus"] = myLabel.fit_transform(data["MaritalStatus"])
data["Over18"] = myLabel.fit_transform(data["Over18"])
data["OverTime"] = myLabel.fit_transform(data["OverTime"])

print(data.head())
print(data.info())

#Now Data is Clean

#train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.drop('Attrition' , axis=1), data['Attrition'], test_size=0.2, random_state=1)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model.fit(X_train, y_train, epochs = 30, batch_size = 30, validation_data = (X_test,y_test),verbose = 1)