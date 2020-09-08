# import libraries
import warnings
warnings.filterwarnings('ignore')

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

import re
import sys
import numpy as np
import pandas as pd
import pickle

from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def load_data(database_filepath):
    '''
    load database saved from ETL

    input: filename and location of database file cretaed from ETL 
    output:
        X = messages
        y = categories (excluding 'related')
        category_names: names of categories that will be used for result labels
    '''
    # create database engine and read files
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('DisasterResponse', con=engine)
    engine.dispose()

    # create X, y
    X = df['message'].values
    y = df.iloc[:,5:].values # exclude 'related', I don't know what this is for

    # create category names for use in main()
    category_names = df.iloc[:, 5:].columns

    return X, y, category_names


def tokenize(text):
    '''
    create lemmatized tokens from input texts
    
    input: text to analyze
    output: tokenized, lemmatized, and cleaned up tokens for use in analysis
    '''
    # remove punctuation, idea cam
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # initiate
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    # remove stop words
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    
    # simplify words, nouns
    clean_tokens = [lemmatizer.lemmatize(tok.lower().strip()) for tok in tokens]
    
    # simplfiy words, verb
    clean_tokens = [lemmatizer.lemmatize(tok, pos='v') for tok in clean_tokens]

    return clean_tokens


def build_model():
    '''
    SVC model for the project; this model perfored best.

    input: none
    output: model for consumption
    '''
    #create pipeline for final model, SVC due to better results
    pipeline = Pipeline([
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),
        
        ('clf', MultiOutputClassifier(OneVsRestClassifier(SVC(C=3,
                                                              kernel='linear', 
                                                              random_state=42))))
    ])

    return pipeline


def evaluate_model(model, X_test, y_test, category_names):
    '''
    print results of models

    input: 
        model: model created from function build_model()
        X_test: test messages
        y_test: test categories
        category_names: names of categories that will be used as labels

    output: classification results
    '''
    # predict
    y_pred = model.predict(X_test)

    #print results
    print(classification_report(y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    '''
    save fitted model to pickle file

    input:
        model: model created from function build_model()
        model_filepath: filename and location of pickle file for further consumption 
    '''
    #export to pickle
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

# ulup0ng