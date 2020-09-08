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
7. [Instructions](#instructions)
7. [Project Web App Link and SCreen Capture](#image)
8. [Related Blogs](#blog)

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
1. A folder that manages ETL named `data` and houses the following data files:<br>
- CSV file containing data: `disaster_categories.csv`
- CSV file containing data: `disaster_messages.csv`
- New sqlite database that houses the transformed data: `DisasterResponse.db`
- A python file for the ETL: `process_data.py`
2. A folder that manages the ML models named `models` and houses the following files:<br>
- A python file that runs the model and outputs a pickle file: `train_classifier.py`
- The pickle file: `classifier.pkl`
3. A folder that manages the web app `app` and houses the following files:<br>
- A pyhton file that initializes Flask: `run.py`
- the `templates` folder conatining go.html and master.html
4. A folder named `notebook` that houses the preparation files for the project
- `ETL_Pipeline_Preparation.ipynb`
- `ML Pipeline Preparation`
5. Icon image file `u1up0ng.png`
6. Screen capture of the web app. `project_image.png`
7. This `README.md` file

### Instructions<a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model<br>
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
2. Run the following command in the app's directory to run the web app.
    `python run.py`

#### Project Web App and Screen Capture<a name="image"></a>
This project lives at [Udacity](https://view6914b2f4-3001.udacity-student-workspaces.com)<br>
![screen_capture](project_image.png?raw=true "screen_capture")

#### Related Blog<a name="blog"></a>
to be completed after this submission

**u1up0ng 2020/09/07**