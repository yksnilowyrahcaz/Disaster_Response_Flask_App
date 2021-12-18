# Disaster Response Pipeline Project
<img src="images/hurricane-ike-2008 PHOTO BY MARK WILSON GETTY IMAGES.jpg" >
Photo: Mark Wilson/Getty Images

## Table of Contents
1. [How To Use This Repository](#howto)
2. [Supporting Packages](#packages)
3. [Project Motivation](#motivation)
4. [About The Dataset](#data)
5. [File Descriptions](#files)
6. [Methodology](#method)
7. [Results](#results)
8. [Licensing, Authors, and Acknowledgements](#licensing)

## How To Use This Repository <a name="howto"></a>
1. Download the zip file of this directory
2. Navigate to this directory on your machine. For the purposes of running the scripts, this will be the root directory.
3. Open the command line from this root directory and run the following commands to set up your database and model.
    - To run the ETL pipeline that cleans data and stores in a database, type the following in the command line:
        `python data/process_data.py data/messages.csv data/categories.csv data/CategorizedMessages.db`
    - To run the ML pipeline that trains classifier and saves, type the following in the command line:
        `python models/train_classifier.py data/CategorizedMessages.db models/classifier.pkl`
4. To run the Flask app, type the following in the command line:
        `python app/run.py`
5. To view the Flask app, open up a browser and go to http://localhost:3001/

## Supporting Packages <a name="packages"></a>
In addition to the standard python libraries, this notebook and analysis rely on the following packages:
- Flask https://flask.palletsprojects.com/en/2.0.x/
- plotly https://plotly.com/
- SQLAlchemy https://www.sqlalchemy.org/
- nltk https://www.nltk.org/
- sklearn https://scikit-learn.org/stable/
- xgboost https://xgboost.ai/

Please see `requirements.txt` for a complete list of packages and dependencies utilized in the making of this project

## Project Motivation <a name="motivation"></a>
The purpose of this repository is to demonstrate the use of an ETL (extract, transform, load) and machine learning pipeline to develop a text classifier deployed using a Flask web application.

The crux of the problem is how to efficiently and effectively interpret communications transmitted during a natural disaster to best respond with the appropriate forms of aid. This remains a challenge because there is typically a large volume of messages that come from social networks and other sources during a natural disaster. Often only a fraction of messages directly relate to a need for assistance and some requests for help are more urgent than others. It is critical that disaster responders can identify the need (food, water, medical aid, electricity) so that the proper aid organizations can be routed to those affected. In sum, the objective is to produce a multi-label classifier that can categorize messages into one or more categories.

## About The Dataset <a name="data"></a>
The messages.csv and categories.csv data files contain 26,248 records representing messages communicated during actual events including an earthquake in Haiti in 2010, an earthquake in Chile in 2010, floods in Pakistan in 2010, super-storm Sandy in the U.S.A. in 2012, and news articles spanning a large number of years and 100s of different disasters. Each message maps to one or more of 36 possible categories categories related to disaster response. Messages are provided in their original language, as well as their English translation and they have been stripped of sensitive information. (Source: [Hugging Face Data Summary](https://huggingface.co/datasets/disaster_response_messages)

This dataset was curated by Appen (formerly Figure Eight). More information about Appen can be found [here](https://appen.com/).

## File Descriptions <a name="files"></a>
| File | Description |
| :--- | :--- |
| data/messages.csv | fields: id, message, original, genre |
| data/categories.csv | fields: id, categories (aid_related, water, etc.) |
| data/etl_pipeline_preparation.ipynb | jupter notebook with code used to develop process_data.py |
| data/process_data.py | etl script that cleans the data for analysis |
| data/CategorizedMessages.db | resulting database from running process_data.py | 
| models/ml_pipeline_preparation.ipynb | jupyter notebook used to develop train_classifier.py |
| models/train_classifier.py | script with machine learning pipeline |
| models/classifier.pkl | pickled (byte serialized) version of the model created by train_classifier.py |
| app/run.py | script that initiates a locally hosted Flask server |
| app/templates/base.html | jinja template used to render the main page of the web app |
| app/templates/go.html | jinja template used to render the classification result of the web app |

## Methodology <a name="method"></a>

### ETL Pipeline
To prepare the messages and categories data for modeling, each file was loaded into a pandas dataframe and an intial pass was made of removing duplicates. Hidden duplicates were identified within the categories data, in which the same id and message occurred more than once with a different set of category labels. This can be seen by using the `.duplicated(keep=False)` method on the categories dataframe. Upon further investigation it was noted that for each hidden duplicate pair, one had one ore more fewer labels that appeared relevant to the message. Thus, judgement was used to choose the duplicate with the most "1" labels, oeprating on the assumption that false negatives are worse than false positives. After filtering out duplicates using this criterion, the categories data was binarized merged with the messages based on common id. Record id # and original (untranslated) messages were dropped from the dataset since they are not used in modeling. Also, the "child_alone" label was dropped because none of the messages were from this category and it did not provide any additional information. The resulting dataframe was saved to a sqlite database.

### Machine Learning Pipeline
Several algorithms were considered in `ml_pipeline_preparation.ipynb`, including sklearn's support vector classifier, multi-layer perceptron (neural network), random forest, and xgboost's boosted random forest classifier. Ultimately, the `XGBRFClassifier` boosted random forest was chosen because it yielded reasonable precision, recall and f1-scores with significantly faster training time. This is in part because decision trees, which ensemble to form random forests, are greedy algorithms that proceed to divide a feature space based on the previous state rather than a global optimum. For this reason trees are efficient. Alone they easily overfit, but as an ensemble, boosted by weighting subsequently selected training samples relative to their residual from the previous tree, better generalization can be achieved.

## Results <a name="results"></a>
<img src="images/Heavy rain poured down in downtown Yangon (Photo-Nay Won Htet).jpg" >
Photo: Nay Won Htet

<img src="images/classification_report.jpg" >

## Licensing, Authors, Acknowledgements <a name="licensing"></a>
Many thanks to Kaggle for administering the survey and all respondents for providing responses. You can find the Licensing for the data and other descriptive information at the Kaggle link available [here](https://www.kaggle.com/c/kaggle-survey-2021). Please feel free to use the code and notebook as you like.
