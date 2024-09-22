# Classification Example
# მეორე ლექცია

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


# CSV ფაილის წაკითხვა.
data = pd.read_csv("https://raw.githubusercontent.com/Athpr123/Binary-Classification-Using-Machine-learning/master/dataset.csv")

# LabelEncoder ინიციალიზაცია.
encoder = LabelEncoder()

# სვეტის წაშლა.
# data.drop('ID', axis=1, inplace=True)
# ვშლით ორ სვეტს. გენდერს იმიტომ რომ 60% იყო მაგ სვეტში NuLL-ები.
data.drop(['ID','Gender'], axis=1, inplace=True)

# ამით ვამოწმებთ რომელი სვეტში, რამდენი NuLL მონაცემია.
# data.isnull().sum()
# პროცენტულობა სვეტში NuLL-ების.
data.isnull().sum() / data.shape[0]

# ვამოწმებთ სვეტ Agency-ში უნიკალური მონაცემები.
data["Agency"].unique()

# სვეტის ფორმატირება. Agency სვეტში მხოლოდ პირველ სიმბოლოებს ვტოვებთ და ხდევბა კლასიფიკაცია პირველი სიმბოლოების მიხედვით.
data["Agency"]=data["Agency"].map(lambda x: x[0])

# ვამოწმევთ სვეტს Agency Type. სადაც გვაქ ორი მონაცემი.
data["Agency Type"].unique()

# ვამოწმებთ ამ სვეტს, რადგანაც გვაქვს ბევრი მონაცემი მოვახდენთ ამ სვეტში ქვეყნების კლასიფიკაციას კონტინეტნტების მიხედვით.
data["Destination"].unique()
def  assign_continent(text):
   if text in ['SINGAPORE','PHILIPPINES','SAUDI ARABIA','THAILAND','HONG KONG','CHINA','BAHRAIN',
               'JAPAN','MALAYSIA','INDONESIA','INDIA','BANGLADESH','NEPAL']:
               return  'Asia'
   elif text in ["CZECH REPUBLIC",'ITALY','GERMANY','GEORGIA','SPAIN']:
        return 'Europe'
   else:
      return 'Other Country'

data['Destination'] = data['Destination'].map(assign_continent)

# Product Name სვეტის კლასიფიკაცია [ 3 ნაწილად ]
data['Product Name'].unique()

def get_product(name):
  if 'Comprehensive' in name:
    return 'Comprehensive'
  elif 'Gold' in name:
    return 'Gold'
  else:
    return 'Other'
data['Product Name'] = data['Product Name'].map(get_product)

# LabelEncoder. string-ები გადაიდს int-ებში. მათემატიკური გამოთვლებისთვის
data['Agency'] = encoder.fit_transform(data['Agency'])
data['Agency Type'] = encoder.fit_transform(data['Agency Type'])
data['Destination'] = encoder.fit_transform(data['Destination'])
data['Distribution Channel'] = encoder.fit_transform(data['Distribution Channel'])
data['Product Name'] = encoder.fit_transform(data['Product Name'])

# მონაცემების დაყოფა
y = data['Claim'].values
X = data.drop(['Claim'], axis=1).values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1) # ძირითადად ეს კოდი default-არის და ხშირად გამოგვადგება ( random_state იცვლება 0 ან 1 )

# Logistic Regression
model = LogisticRegression(max_iter=45000)
model.fit(X_train,y_train)
model.score(X_test,y_test)

# ჩარტები

fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1], pos_label=1) # [:1] -> 1 იანის მოსვლის ალბათობა
plt.plot(fpr,tpr)
plt.show()

print(data.head())

# ერორების გასაფიქსად ზედი გვაქ და მერე იქიდან გადავაკეთოთ და შევამოწმოთ.
