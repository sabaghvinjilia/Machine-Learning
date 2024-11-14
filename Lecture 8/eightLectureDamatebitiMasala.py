import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

data = pd.read_csv('https://raw.githubusercontent.com/dmgutierrez/Sarcasm-detector/refs/heads/master/data/train.csv')
print("Before")
print(data.head())
stopset = set(stopwords.words("english"))
wnl = WordNetLemmatizer()

def clean_text(s):

  text = [wnl.lemmatize(w).lower().strip() for w in word_tokenize(s) if not (w in stopset) and not (w in string.punctuation) and not w.isdigit()]
  text = " ".join(text)
  return text

data['text'] = data['text'].map(clean_text)

print("After")
print(data.head())