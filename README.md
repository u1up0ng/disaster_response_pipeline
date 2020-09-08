![ulupong](ulupong.png?raw=true "ulupong")

<h1><center>Udacity: Disaster Response Project</center></h1>


### Project: Disaster Response Pipeline Web App

### Table of Contents
1. [Project Motivation](#pm)
2. [Project Goal](#goal)
3. [Notes](#notes)
4. [Data](#data)
5. [Install](#install)
6. [Contents](#contents)
7. [Related Blogs](#blog)

#### Project Motivation<a name="pm"></a>
This is repository contains project codes required to complete the Data Engineering Section of Udacity's Data Scientist nano degree.

#### What is this project trying to achieve?<a name="goal"></a>
The final deliverable of the project is a web app that employs ETL, NLP, and Machine Learning Pipeline lessons taught from the nano degree.<br>

#### Notes<a name="notes"></a>
C-Support Vector Classification was the model selected for the project due to higher score achieved.

#### Data <a name="data"></a>
Two CSV files holds the data, courtesy of [Figure Eight](https://www.welcome.ai/figure-eight). Files 'disaster_categories.csv' and 'disaster_messages.csv', are pre-formatted data, that needed ETL for it to be loaded to a sqlite database.

#### Install<a name="install"></a>
This project requires Python and the following Python libraries installed:

- numPy
- pandas
- matplotlib
- scikit-learn
- pickle
- sqlalchemy
- nltk
- re
- sys
- plotly
- json
- flask
- joblib


If you do not have Python installed yet, it is highly recommended that you install the Anaconda distribution of Python, which already have most of the above packages.

#### Content<a name="contents"></a>
The archive submission includes the following files and directories:<br>
1. A folder that manages ETL named 'data' and houses the following files:<br>
- CSV dile: disaster_categories.csv
- CSV file: disaster_messages.csv
- sqlite database: DisasterResponse.db
- python file: process_data.py
2. A folder that manages the ML models named 'models' and houses the following files:<br>
- python file: train_classifier.py
- pickle file: classifier.pkl
3. A folder that manages the web app 'app' and houses the following files:<br>
- pyhton file: run.py
- the 'templates' folder conatining go.html and master.html
4. A folder named 'notebook' that houses the preparation files for the project
- ETL_Pipeline_Preparation.ipynb
- ML Pipeline Preparation
5. Icon image file (u1up0ng.png)
6. This README.md file


#### Related Blog<a name="blog"></a>
to be completed after this submission

**u1up0ng 2020/09/07**