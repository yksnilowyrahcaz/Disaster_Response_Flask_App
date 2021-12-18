# Disaster Response Pipeline Project

<img src="images/hurricane-ike-2008 PHOTO BY MARK WILSON GETTY IMAGES.jpg" >
Photo: Mark Wilson/Getty Images

## Table of Contents

1. [How To Use This Repository](#howto)
2. [Supporting Packages](#packages)
3. [Project Motivation](#motivation)
4. [About The Data][(#data)
5. [File Descriptions](#files)
6. [Results](#results)
7. [Licensing, Authors, and Acknowledgements](#licensing)

## How To Use This Repository <a name="howto"></a>

1. Download the zip file of this directory
2. Navigate to this directory on your machine. For the purposes of running the scripts, this will be the root directory.
3. Open the command line from this root directory and run the following commands to set up your database and model.
    - To run the ETL pipeline that cleans data and stores in a database, type the following in the command line:
        `python data/process_data.py data/messages.csv data/categories.csv data/CategorizedMessages.db`
    - To run the ML pipeline that trains classifier and saves, type the following in the command line:
        `python models/train_classifier.py data/CategorizedMessages.db models/classifier.pkl`
4. To run the Flask app, type the following in the command line:
        `python app/run.py
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

The theme of the project is centered around the open problem of how to efficiently and effectively interpret communications transmitted during a natural disaster to best respond with the appropriate forms of aid. This remains a challenge because there is typically a large volume of messages that come through social networks and other forms of media. Often only a fraction of messages directly relate to an identifiable need and some requests for help are more urgent than others. It is critical that disaster responders can identify the need (food, water, medical aid, electricity) so that the proper aid organizations can be routed to those affected.

In sum, a multi-label classifier is needed to identify messages corresponding to one or more categories. To that end, we have developed an ETL pipeline that cleans 

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

Included is a notebook available here to showcase work related to the above questions. Markdown cells are used to walk the reader through the analysis performed. The raw survey response data used in this analysis is openly available on Kaggle available [here](https://www.kaggle.com/c/kaggle-survey-2021/data).

## Results <a name="results"></a>
<img src="images/Heavy rain poured down in downtown Yangon (Photo-Nay Won Htet).jpg" >
Photo: Nay Won Htet

The main findings of this analysis can be found at the post available [here](https://medium.com/@zacharywolinsky/this-new-data-will-make-you-rethink-your-role-in-accounting-finance-8d2f25262440).

## Licensing, Authors, Acknowledgements <a name="licensing"></a>
Many thanks to Kaggle for administering the survey and all respondents for providing responses. You can find the Licensing for the data and other descriptive information at the Kaggle link available [here](https://www.kaggle.com/c/kaggle-survey-2021). Please feel free to use the code and notebook as you like.
