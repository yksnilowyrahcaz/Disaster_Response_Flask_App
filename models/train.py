# import libraries
import re
import sys
import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from xgboost import XGBRFClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(database_filepath):
    '''
    Load the cleaned data from the database
    
    Parameters
        database_filepath: str, the file path of the database file
        
    Returns
        X: numpy.ndarray, containing the message strings
        Y: numpy.ndarray, containing the target categories
        category_names: list, of 35 possible categories
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table(f'{database_filepath}', con=engine)
    X = df.message.values
    Y = df.iloc[:, 2:].values
    category_names = list(df.iloc[:,2:].columns)
    
    return X, Y, category_names

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

def build_model():
    '''
    Construct a Pipeline instance that vectorizes text and performs multi-label classification
    
    Returns
        pipeline: sklearn.pipeline.Pipeline, includes transformer and estimator
    '''
    
    pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=tokenize, max_features=10000, n_gram_range=(1,2))),
    ('clf', MultiOutputClassifier(XGBRFClassifier(use_label_encoder=False, verbosity=0, learning_rate=0.1)))
])
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Predicts labels based on test data and reports prediction accuracy, precision, recall, f1-score
    
    Parameters
        model: estimator, in this case the pipeline constructed from build_model()
        X_test: numpy.ndarray, containing the message strings from the test set
        Y_test: numpy.ndarray, containing the target categories from the test set
        category_names: list, of 35 possible categories
    
    Returns a print out of the model accuracy and classification report,
        which includes precision, recall, f1-score metrics
    '''
    
    Y_pred = model.predict(X_test)
    print('Model Accuracy: ', (Y_test == Y_pred).mean())
    print(classification_report(y_test, y_pred, target_names=category_names))

def save_model(model, model_filepath):
    '''
    Saves the learned model to a pickle file (.pkl)
    
    Parameters
        model: estimator, in this case the pipeline constructed from build_model()
        model_filepath: str, the file path where the pickled model will be saved
    '''
    
    pickle.dump(model, open(f'{model_filepath}.pkl', 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print(f'Loading data...\n\tDATABASE: {database_filepath}')
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1729)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(X_test, Y_test, category_names)

        print(f'Saving model...\n\tMODEL: {model_filepath}')
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()