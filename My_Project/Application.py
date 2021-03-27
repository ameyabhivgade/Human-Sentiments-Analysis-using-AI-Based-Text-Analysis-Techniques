# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 18:21:17 2021

@author: AMEYA
"""

import pickle
import pandas as pd
import webbrowser
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.express as px
import sqlite3 as sql
from dash.dependencies import Input, Output, State
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

# Declaring Global variables
# A variable declared outside a function is a global variable by default.

app = dash.Dash(external_stylesheets=[dbc.themes.CYBORG])
project_name = "Understanding Human Sentiments Using AI based Text Analysis Techniques"

def load_model():
    global pickle_model
    global vocab
    global scrappedReviews
    
    conn = sql.connect('Review_db.db')
    scrappedReviews = pd.read_sql('SELECT reviews FROM scrappedreviews', conn)
    scrappedReviews = scrappedReviews['reviews'].values.tolist()
    
    file = open("pickle_model.pkl", 'rb') 
    pickle_model = pickle.load(file)

    file = open("features.pkl", 'rb') 
    vocab = pickle.load(file)

def open_browser():
    # Open the default web browser
    webbrowser.open_new('http://127.0.0.1:8050/')
    
def generate_pie_chart():
    global fig
    dataframe_1 = pd.read_csv('balanced_reviews.csv')
    labels = ['Negative Reviews','Positive Reviews']
    preds = dataframe_1['Positivity'].tolist()
    positive_count,negative_count = preds.count(1),preds.count(0)
    values = [negative_count,positive_count]
    fig = px.pie(values = values,
                 names = labels,
                 title='Balanced Review Distribution of Data to Train Model',
                 width=1000,height=500,hole=0.25,
                 template="plotly_dark",
                 color_discrete_sequence=['FireBrick','Chartreuse'])
    

        
def check_review(reviewText):
    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=vocab)
    reviewText = transformer.fit_transform(loaded_vec.fit_transform([reviewText]))
    return pickle_model.predict(reviewText)

def create_app_ui():
    global project_name
    main_layout = dbc.Container(
        dbc.Jumbotron(  
                [   
                    html.H2(id = 'heading', children = project_name, className = 'display-6 mb-7',
                            style={'text-align':'center','color':'White',}),
                    dbc.Container([dcc.Graph(figure=fig)],style={'margin-bottom':'30px','margin-top':'30px','width': '1020px'}),
                    dbc.Container([dcc.Dropdown(
                                        id='dropdown',
                                        placeholder = 'Select a Review',
                                        options=[{'label': i[:100] + "...", 'value': i} for i in scrappedReviews],
                                        value = scrappedReviews[0],
                                        style = {'margin-bottom': '35px','min-width':'1000px','padding-top':'10px'}
                    
                                            )
                                    ],
                        style = {'padding-left': '18px', 'padding-right': '25px'}
                        ),
                    dbc.Button("Submit", color="dark", className="mt 2 mb-3", id = 'button', style = {'width': '1000px'}),                  
                    html.Div(id = 'result1'),
                    
                    dbc.Textarea(id = 'textarea', 
                                 className="mb-3", 
                                 placeholder="Enter the Review", 
                                 value = 'Start Typing ....', 
                                 style = {'height': '150px','margin-top': '30px'}),
                    dbc.Button("Submit", color="dark", className="mt-2 mb-3", id = 'button1', 
                               style = {'width': '1000px'}),
                    html.Div(id = 'result')
                    
                    ],
                className = 'text-center'
                ),
        className = 'mt-4'
        )
    
    return main_layout

@app.callback(
    Output('result1', 'children'),
    [
    Input('button', 'n_clicks')
    ],
    [
     State('dropdown', 'value')
     ]
    )
def update_dropdown(n_clicks, value):
    result_list = check_review(value)
    
    if (result_list[0] == 0 ):
        return dbc.Alert("The Review is Negative", color="danger")
    elif (result_list[0] == 1 ):
        return dbc.Alert("The Review is Positive", color="success")
    else:
        return dbc.Alert("Unknown", color="dark")
    
@app.callback(
    Output('result', 'children'),
    [
    Input('button1', 'n_clicks')
    ],
    [
    State('textarea', 'value')
    ]
    )    
def update_app_ui(n_clicks, textarea):
    result_list = check_review(textarea)
    
    if (result_list[0] == 0 ):
        return dbc.Alert("The Review is Negetive", color="danger")
    elif (result_list[0] == 1 ):
        return dbc.Alert("The Review is Positive", color="success")
    else:
        return dbc.Alert("Unknown", color="dark")
    
    
    
def main():
    global app
    global project_name
    generate_pie_chart()
    load_model()
    open_browser()
    app.layout = create_app_ui()
    app.title = project_name
    app.run_server()
    app = None
    project_name = None
	
if __name__ == '__main__':
    main()

