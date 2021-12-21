import sys
from pathlib import Path
sys.path.insert(0,'\\'.join(Path.cwd().as_posix().split('/')))

import re
import json
import joblib
import pandas as pd
import plotly
import plotly.express as px

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from models.train_classifier import tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
from sqlalchemy import create_engine

app = Flask(__name__)

# load data
engine = create_engine('sqlite:///data/CategorizedMessages.db')
df = pd.read_sql_table('CategorizedMessages', engine)

# load model
model = joblib.load('models/classifier.pkl')

@app.route('/')
@app.route('/base')
def base():
    '''
    base webpage to display visuals and receive user input text for model
    '''
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message'].reset_index()
    category_counts = df.iloc[:,2:].sum().reset_index().sort_values(by=0)
    category_counts.columns = ['category','count']
    
    # create figures for the visualizations
    fig1 = px.pie(genre_counts, names='genre', values='message', labels='genre',
                  title='Composition of the dataset by message genre', hole=0.3,
                  color_discrete_sequence=px.colors.qualitative.Dark2_r, 
                  width=500, height=500).update_traces(marker_line_color='rgb(0,0,0)',
                                                       hovertemplate='Genre: %{label}',
                                                       textfont_size=15,
                                                       marker_line_width=1)

    fig2 = px.bar(category_counts, x='count', y='category', color='count',
                  color_continuous_scale= px.colors.sequential.Aggrnyl_r,
                  title='Composition of the dataset by category', text='count',
                  hover_data={'count':':,.0f'},orientation='h', range_x=[0,22000], 
                  width=700, height=700).update_traces(marker_line_color='rgb(0,0,0)', 
                                                       marker_line_width=1,
                                                       texttemplate='%{text:,.0f}',
                                                       hovertemplate='Category: %{y}'+
                                                       '<br>Count: %{text:,.0f}</br>',
                                                       textposition='outside')
    # encode plotly graphs in JSON
    graphs = [fig1, fig2]
    ids = [f'graph-{i}' for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('base.html', ids=ids, graphJSON=graphJSON)

@app.route('/go')
def go():
    '''
    web page that handles user query and displays model results
    '''
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[2:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
        app.run(host='0.0.0.0', port=3001, debug=False)

if __name__ == '__main__':
    main()