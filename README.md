# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://localhost:3001/

<img src="images/knowledge_wide.jpg" >
Stuart Kinlough/Getty Images/Ikon Images

## Table of Contents

1. [Supporting Packages](#packages)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Supporting Packages <a name="packages"></a>
In addition to the standard python libraries, this notebook and analysis rely on the following packages:
- geopandas https://geopandas.org/getting_started.html
- GDAL http://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal
- Fiona http://www.lfd.uci.edu/~gohlke/pythonlibs/#fiona
- pyproj http://www.lfd.uci.edu/~gohlke/pythonlibs/#pyproj
- rtree http://www.lfd.uci.edu/~gohlke/pythonlibs/#rtree
- shapely http://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely
- mapclassify https://pysal.org/mapclassify/
- seaborn https://seaborn.pydata.org/

## Project Motivation <a name="motivation"></a>
The purpose of this repository is to provide an example of exploratory data analysis (EDA) in the context of the 2021 Kaggle Machine Learning and Data Science Survey. My goal is to better understand data scientists in the Accounting/Finance industry by exploring the following questions:

1. Which countries had respondents from the Accounting/Finance industry? How do these countries rank in their respective percentage of respondents who work in the Accounting/Finance industry?
2. What ML and DS roles do accounting and finance professionals work in? What programming languages do they prefer and what algorithms are most popular?
3. What is the gender profile of all survey respondents? How does the Accounting/Finance industry rank among all other industries in the percentage of Data Scientists who identify as woman, non-binary or another gender descriptor?

## File Descriptions <a name="files"></a>
Included is a notebook available here to showcase work related to the above questions. Markdown cells are used to walk the reader through the analysis performed. The raw survey response data used in this analysis is openly available on Kaggle available [here](https://www.kaggle.com/c/kaggle-survey-2021/data).

## Results <a name="results"></a>
<img src="images/map_acct.jpg" >

The main findings of this analysis can be found at the post available [here](https://medium.com/@zacharywolinsky/this-new-data-will-make-you-rethink-your-role-in-accounting-finance-8d2f25262440).

## Licensing, Authors, Acknowledgements <a name="licensing"></a>
Many thanks to Kaggle for administering the survey and all respondents for providing responses. You can find the Licensing for the data and other descriptive information at the Kaggle link available [here](https://www.kaggle.com/c/kaggle-survey-2021). Please feel free to use the code and notebook as you like.
