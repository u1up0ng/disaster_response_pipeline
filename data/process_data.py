# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sys


def load_data(messages_filepath, categories_filepath):
  '''
  load csv file for both messages and categories

  input:
    messages_filepath   : file path for disaster_messages.csv
    categories_filepath : file path for disaster_categories.csv
  output:
    df                  : base dataframe  
  '''

  # load messages dataset
  messages = pd.read_csv(messages_filepath)
  # load categories dataset
  categories = pd.read_csv(categories_filepath)
  # merge datasets
  df = messages.merge(categories, on='id')

  return df


def clean_data(df):
  '''
  creates the working dataframe containing categories for the messages

  input                 : base dataframe
  output                : paroject dataframe

  '''

  # create a dataframe of the 36 individual category columns
  categories = df.categories.str.split(';', expand=True)
  # select the first row of the categories dataframe
  row = categories.iloc[0]
  # use this row to extract a list of new column names for categories.
  category_colnames = row.apply(lambda x: x[:-2])
  # rename the columns of `categories`
  categories.columns = category_colnames

  # convert category values to just numbers 0 or 1
  for column in categories:
    # set value to be the last character of the string and convert to numeric    
    categories[column] = categories[column].str[-1].astype(np.int)

  # drop the original categories column from `df`
  df.drop('categories', axis=1, inplace=True)
  # concatenate the original dataframe with the new `categories` dataframe
  df = pd.concat([df, categories], axis=1)

  # drop duplicates
  df = df.drop_duplicates()

  return df

def save_data(df, database_filename):
  '''
  saves project dataframe to sqlite

  input:
    df                  : project dataframe
    database_filenabe   : path and filename of project database
  '''

  engine = create_engine(f'sqlite:///{database_filename}')
  df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')
  # close engine
  engine.dispose()


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()

# u1up0ng 20200831