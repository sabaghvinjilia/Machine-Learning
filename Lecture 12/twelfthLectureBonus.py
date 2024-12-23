from transformers import pipeline
from bs4 import BeautifulSoup
import requests
from deep_translator import GoogleTranslator

model = pipeline('sentiment-analysis')
# print(model('saba likes machine learning')[0]['label'])
print("Nika Is Nigger", model('Nika Is Nigger')[0]['label'])
print("Nika Is White", model('Nika Is White')[0]['label'])

model1 = GoogleTranslator(source='auto', target='en')
url='https://www.booking.com/hotel/ge/senakshi-saojaxo-sastumro.en-gb.html?aid=356980&label=gog235jc-1FCAsoUkIZc2VuYWtzaGktc2FvamF4by1zYXN0dW1yb0gzWANoUogBAZgBCbgBB8gBDNgBAegBAfgBDIgCAagCA7gClaiRuwbAAgHSAiQ4ODA5Yjk1ZS1mMWNlLTQ4ZWMtYWI2Yy1jMzQ5YzEyM2VkNDXYAgbgAgE&sid=3a9d4e132046e16fc51e3a2fbdcbc132&age=0&checkin=2024-12-19&checkout=2024-12-20&dest_id=-2329739&dest_type=city&dist=0&group_adults=5&group_children=1&hapos=1&hpos=1&no_rooms=1&req_adults=5&req_age=0&req_children=1&room1=A%2CA%2CA%2CA%2CA%2C0&sb_price_type=total&soh=1&sr_order=popularity&srepoch=1734628382&srpvid=706e790bbc4a04a5&type=total&ucfs=1&#no_availability_msg'
content = requests.get(url).text
content = BeautifulSoup(content, 'html.parser')
for text in content.find_all('div',class_='a53cbfa6de b5726afd0b'):
    print(model1.translate(text.text), "has sentiment", model(model1.translate(text.text))[0]['label'])