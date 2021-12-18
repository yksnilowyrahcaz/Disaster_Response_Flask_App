# Disaster Response Pipeline Project

<img src="images/hurricane-ike-2008 PHOTO BY MARK WILSON GETTY IMAGES.jpg" >
Photo: Mark Wilson/Getty Images

## Table of Contents

1. [How To Use This Repository](#howto)
2. [Project Motivation](#motivation)
3. [Supporting Packages](#packages)
4. [File Descriptions](#files)
5. [Results](#results)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## How To Use This Repository <a name="howto"></a>

1. Download the zip file of this directory
2. Navigate to this directory on your machine. For the purposes of running the scripts, this will be the root directory.
3. Open the command line from this root directory and run the following commands to set up your database and model.
    - To run the ETL pipeline that cleans data and stores in a database, type the following in the command line:
        `python data/process_data.py data/messages.csv data/categories.csv data/CategorizedMessages.db`
    - To run the ML pipeline that trains classifier and saves, type the following in the command line:
        `python models/train_classifier.py data/CategorizedMessages.db models/classifier.pkl`
    - To run the Flask app, type the following in the command line:
        `python app/run.py
4. To view the Flask app, open up a browser and go to http://localhost:3001/

## Project Motivation <a name="motivation"></a>
The purpose of this repository is to demonstrate the use of an ETL (extract, transform, load) and machine learning pipeline in the deployment of a language model using a Flask web application.

The theme of the project is centered around the open problem of how to efficiently interpret communications transmitted during a natural disaster to best respond with the appropriate forms of aid. This remains a challenging task because there is typically a large volume of messages that come through social networks and other forms of media and often only a fraction of them directly relate to an immediate need. It is critical that disaster responders know what this need is (food, water, medical aid, electricity) so that the proper aid organizations can be routed to those affected.

## Supporting Packages <a name="packages"></a>
In addition to the standard python libraries, this notebook and analysis rely on the following packages:
- Flask https://flask.palletsprojects.com/en/2.0.x/
- plotly https://plotly.com/
- SQLAlchemy https://www.sqlalchemy.org/
- nltk https://www.nltk.org/
- sklearn https://scikit-learn.org/stable/
- xgboost https://xgboost.ai/

Please see `requirements.txt` for a complete list of packages and dependencies utilized in the making of this project

## File Descriptions <a name="files"></a>
Included is a notebook available here to showcase work related to the above questions. Markdown cells are used to walk the reader through the analysis performed. The raw survey response data used in this analysis is openly available on Kaggle available [here](https://www.kaggle.com/c/kaggle-survey-2021/data).

## Results <a name="results"></a>
<img src="images/Heavy rain poured down in downtown Yangon (Photo-Nay Won Htet).jpg" >
Photo: Nay Won Htet

The main findings of this analysis can be found at the post available [here](https://medium.com/@zacharywolinsky/this-new-data-will-make-you-rethink-your-role-in-accounting-finance-8d2f25262440).

## Licensing, Authors, Acknowledgements <a name="licensing"></a>
Many thanks to Kaggle for administering the survey and all respondents for providing responses. You can find the Licensing for the data and other descriptive information at the Kaggle link available [here](https://www.kaggle.com/c/kaggle-survey-2021). Please feel free to use the code and notebook as you like.
