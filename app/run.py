import re
import json
import plotly
import joblib
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine

app = Flask(__name__)

def tokenize(text):
    '''
    Convert a string of text into a list of tokens
    
    Parameters
        text: str, containing the text to be tokenized
        
    Returns
        clean_tokens: list, containing normalized, tokenized and lemmatized text
    '''
    # detect and replace urls with placeholder
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')

    # case normalize and remove punctuation, except for $ sign
    text = re.sub(r'[^a-zA-Z$]', ' ', text.lower())
    
    # tokenize
    tokens = word_tokenize(text)
    
    # remove stop words and lemmatize
    lem = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    clean_tokens = [lem.lemmatize(tok).strip() for tok in tokens if tok not in stop_words]

    return clean_tokens

# load data
engine = create_engine('sqlite:///data/CategorizedMessages.db')
df = pd.read_sql_table('CategorizedMessages', engine)

# load model
model = joblib.load('models/classifier.pkl')

@app.route('/')
@app.route('/index')
def index():
    '''
    index webpage displays cool visuals and receives user input text for model
    '''
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = [f'graph-{i}' for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('index.html', ids=ids, graphJSON=graphJSON)

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