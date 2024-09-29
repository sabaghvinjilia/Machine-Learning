# Linear Regression
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def get_Sales(Tv):
    sales = model.predict([[Tv,Tv*Tv]])
    return f'predicted Sale is : {sales}'

poly = PolynomialFeatures( degree=2 , include_bias=False )
model = LinearRegression()
data = pd.read_csv("TV Marketing.csv")
data = data.sort_values(by='TV')
print(data.head())
X = data['TV'].values
y = data['Sales'].values
X = X.reshape(-1,1)
X = poly.fit_transform(X)
print(X)
model.fit(X,y)
print(model.score(X,y))

# G radio support funciton (amis gareshe gradio ar imusahvebs) + funqcia swirdeba -> get_Sales
# Web Inteface
TV = gr.Number( label='Enter your TV' )
Sales = gr.Textbox( label='Result is' )
myInterface = gr.Interface( fn=get_Sales, inputs=TV, outputs=Sales).launch()