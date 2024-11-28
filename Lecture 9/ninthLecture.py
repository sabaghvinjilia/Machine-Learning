import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from bs4 import BeautifulSoup
import requests

from deep_translator import GoogleTranslator

movies = pd.read_csv('imdb_movies.csv', encoding='latin-1')

movies.columns = [ 'text', 'label']

movies['text'] = movies['text'].map(lambda x : re.sub(r'\d+',"",x))
movies['text'] = movies['text'].map(lambda x: re.sub(r'[\([{})\]]',"",x))

# print(movies.loc[0,'text'])
# print(movies.head())

X = movies['text']
y = movies['label']

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2, random_state=1)

tranformer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train = tranformer.fit_transform(X_train).toarray()
X_test = tranformer.fit_transform(X_test).toarray()

# print(movies.head())
# print(X_train)

model = RandomForestClassifier()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))

url ='https://sportall.ge/fekhburthi/fekhburthii/legionerebi/174649-ra-ushlis-saqarthvelos-nakrebs-khels-ufro-didi-miznebisken-sad-varth-obieqturad.html?orderBy=0&all=1&add_new=0&reply=0#comments'
content = requests.get(url).text
content = BeautifulSoup(content,'html.parser')

comments = content.find_all("div",class_ ="c_comment")
print(comments)

translator = GoogleTranslator(source='auto', target='en')

for comment in comments:
  english_version =translator.translate(comment.text)
  label = model.predict(transformer.transform([english_version]))[0]
  print(label)

movies.loc[0,'text']