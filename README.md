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
8. [Sources & References](#sources)

## How To Use This Repository <a name="howto"></a>
The flask app provided in this repository is separately deployed using [Heroku](https://www.heroku.com/home) and can be found [here](https://disaster-response-flask-webapp.herokuapp.com/). This can be recreated and run on your local machine by completing the following steps:

1. Download and unzip this repository to your local machine.
2. Navigate to this directory and open the command line. For the purposes of running the scripts, this will be the root directory.
3. Create a virtual environment to store the supporting packages

        python -m venv ./venv

4. Activate the virtual environment

        .\venv\Scripts\activate

5. Install the supporting packages from the requirements.txt file

        pip install -r requirements.txt
        
6. (Optional) To run the ETL pipeline that cleans data and stores in a database, type the following in the command line:
        
        python data/process_data.py data/messages.csv data/categories.csv data/CategorizedMessages.db
    
Note: CategorizedMessages.db is already provided within this repo and does not need to be recreated to run the flask app at step 8.

7. (Optional) To run the ML pipeline that trains classifier and saves, type the following in the command line:
       
        python models/train_classifier.py data/CategorizedMessages.db models/classifier.pkl
       
Note: training the model may take a significant amount of time. The pickled model is already included in this repository and you can proceed to step 8 without retraining the model.
       
8. To run the Flask app, type the following in the command line:
       
       python app/run.py
       
9. To view the Flask app being served on your local machine, open a browser, and go to http://localhost:3001/ if you are using Windows.

## Supporting Packages <a name="packages"></a>
In addition to the standard python libraries, this notebook and analysis rely on the following packages:
- [Flask](https://flask.palletsprojects.com/en/2.0.x/)
- [Plotly](https://plotly.com/)
- [SQLAlchemy](https://www.sqlalchemy.org/)
- [NLTK](https://www.nltk.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [XGBoost](https://xgboost.ai/)

Please see `requirements.txt` for a complete list of packages and dependencies utilized in the making of this project

## Project Motivation <a name="motivation"></a>
The purpose of this repository is to demonstrate the use of an ETL (extract, transform, load) and machine learning pipeline to develop a text classifier deployed using a Flask web application.

The crux of the problem is how to efficiently and effectively interpret communications transmitted during a natural disaster to best respond with the appropriate forms of aid. This remains a challenge because there is typically a large volume of messages that come from social networks and other sources during a natural disaster. Often only a fraction of messages directly relate to a need for assistance and some requests for help are more urgent than others. It is critical that disaster responders can identify the need (food, water, medical aid, electricity) so that the proper aid organizations can be routed to those affected. In sum, the objective is to produce a multi-label classifier that can categorize messages into one or more categories.

## About The Dataset <a name="data"></a>

<img src="images/overview_training_data.jpg" >

The messages.csv and categories.csv data files contain 26,248 records representing messages communicated during actual events including an earthquake in Haiti in 2010, an earthquake in Chile in 2010, floods in Pakistan in 2010, super-storm Sandy in the U.S.A. in 2012, and news articles spanning many years and 100s of different disasters. Each message maps to one or more of 36 possible categories related to disaster response. Note: the "child_alone" category was removed from training because none of the messages had this label, thus no new information was gained from modeling the messages on this label. Messages are provided in their original language, as well as their English translation and they have been stripped of sensitive information. (Source: [Hugging Face Data Summary](https://huggingface.co/datasets/disaster_response_messages))

## File Descriptions <a name="files"></a>
| File | Description |
| :--- | :--- |
| data/messages.csv | fields: id, message, original, genre |
| data/categories.csv | fields: id, categories (aid_related, water, etc.) |
| data/etl_pipeline_preparation.ipynb | jupyter notebook with code used to develop process_data.py |
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
To prepare the messages and categories data for modeling, each file was loaded into a pandas dataframe and an initial pass was made of removing duplicates. Hidden duplicates were identified within the categories data, in which the same id and message occurred more than once with a different set of category labels. This can be seen by executing `categories.id.duplicated(keep=False)` after the initial duplicates removal.

Upon further investigation it was noted that for each hidden duplicate pair, one had one or more fewer labels that appeared relevant to the message. Judgement was used to choose the duplicate with the most "1" labels, operating on the assumption that false negatives are worse than false positives. 

After filtering out duplicates using this criterion, the categories data was binarized and merged with the messages based on common id. Record id # and original (untranslated) messages were dropped from the dataset since they are not used in modeling. Also, the "child_alone" label was dropped because none of the messages were from this category and it did not provide any additional information. The resulting dataframe was saved to a sqlite database.

### Machine Learning Pipeline
Several algorithms were considered in `ml_pipeline_preparation.ipynb`, including sklearn's support vector classifier, multi-layer perceptron (neural network), random forest, and xgboost's boosted random forest classifier. Ultimately, the `XGBRFClassifier` boosted random forest was chosen because it yielded reasonable precision, recall and f1-scores with significantly faster training time. 

Decision trees, which ensemble to form random forests, are greedy algorithms that proceed to divide a feature space based on the previous state rather than a global optimum. For this reason, trees are efficient. Alone they easily overfit, but as an ensemble, boosted by weighting subsequently selected training samples relative to their residual from the previous tree, better generalization can be achieved.

Using sklearn's `GridSearchCV` class, a search over various combinations of `ngram_range` and `max_features` parameters of the `TfidfVectorizer`, as well as the `learning_rate` parameter of the `XGBRFClassifer`, was performed with 5-fold cross-validation. It was initially noted that slight gains in performance were achieved with `ngram_range=((1,2))`, `max_features=10000`, and `learning_rate=0.1` parameter values. However, upon exploring grid search with other parameters, performance appeared to be improved more dramatically by adjusting tree depth, number of estimators, the percentage of features randomly selected for each level, and the max number of features used from the tfidf matrix. Details discussed in the following section.

## Results <a name="results"></a>
The final model learned using the boosted random forest algorithm yielded a weighted average score of 73% precision, 62% recall and 65% F1-score (harmonic mean of precision and recall). Precision is the proportion of positive samples predicted by the model that are in fact positive samples. Recall is the proportion of true positive samples are correctly predicted by the model. Accuracy is the proportion of predictions, positive and negative, that are correctly classified by the model. 

Note that although the model's final accuracy happened to be ~95%, accuracy as a performance metric can be misleading because the dataset used to train on is imbalanced. For example, there are many more messages labeled as "aid_related" than there are for messages labeled as "water", and fewer yet that are labeled as "search_and_rescue". Precision and recall can be more helpful metrics in this context.

Here are the results from grid search. After trying out various parameter combinations, the depth of the trees (`max_depth` parameter) and the number of estimators (`n_estimators` parameter) of the `XGBRFClassifier` class appeared to have a significant impact on model performance on both training and test sets. `XGBRFClassifier`'s `colsample_bylevel` parameter and `TfidfVectorizer`'s `max_features` parameter also appeared to be effective levers for balancing model complexity and underfitting.

<p align="center">
<img src="images/cv_results.jpg" >
</p>

The following shows the parameters of the final model that is deployed, with tuned parameters highlighted in green:

<p align="center">
<img src="images/final_model.jpg" >
</p>

And here is the resulting classification report:

<p align="center">
<img src="images/classification_report.jpg" >
</p>

The image below shows the output of the app after entering the following message: 

> #### _"If you receive this message, please help as soon as you can because a power line went down during the wind storm and many of us need medical care and we are desperately thirsty"_

Notice that the model appropriately maps "power line" to electricity and "thirsty" to water:

<img src="images/classify_1.jpg" >

<img src="images/classify_2.jpg" >

<img src="images/classify_3.jpg" >

## Sources & References <a name="sources"></a>
The data used in this project were curated by [Appen (formerly Figure Eight)](https://appen.com/) and are openly available for use. The dataset and much of the supporting html was obtained through the [Udacity Data Scientist Nanodegree Program](https://www.udacity.com/blog/2018/05/introducing-udacity-data-scientist-nanodegree-program.html). Rahul Pandey's blog post, ["A simple guide to deploy machine learning model on Heroku"](https://medium.com/analytics-vidhya/a-simple-guide-to-deploy-machine-learning-model-on-heroku-4ccbae6d2b7b) was helpful for getting the project onto the internet. Please feel free to use the contents of this repository for your own projects.
