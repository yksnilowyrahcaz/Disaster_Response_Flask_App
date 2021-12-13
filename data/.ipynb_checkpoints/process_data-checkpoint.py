import re
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def ETL(messages_filename, categories_filename, database_filename):
    '''
    Extract, transform and load the messages and categories data into a sqlite database
    
    Parameters
    ----------
        messages_filename: str
            name of existing messages dataset file
            
        categories_filename: str
            name of existing categories dataset file
            
        database_filename: str
            desired name of the output sqlite database
    '''
    
    # load messages and categories data
    print(f'Loading data from...\n\tMESSAGES: {messages_filename}\n\tCATEGORIES: {categories_filename}')
    messages = pd.read_csv(messages_filename)
    categories = pd.read_csv(categories_filename)
    
    # remove duplicates
    print('Cleaning data...')
    messages.drop_duplicates(inplace=True)
    categories.drop_duplicates(inplace=True)
    
    # remove hidden duplicates having same message, but different category counts
    count_cats = lambda x: sum([min(int(i),1) for i in re.findall('[0-9]', x)])
    categories['cat_counts'] = categories.categories.apply(count_cats)
    keep_dict = categories.groupby('id').cat_counts.max().to_dict()
    index = categories.id.apply(lambda x: keep_dict[x]) == categories.cat_counts
    categories = categories.loc[index].drop('cat_counts', axis=1)

    # remove hidden duplicates having same category count, but different category labels
    categories.drop_duplicates(subset='id', inplace=True)
    
    # create a dataframe of the 36 individual category columns
    categories_expanded = categories.drop('id', axis=1)['categories'].str.split(pat=';', expand=True)
    
    # rename columns
    categories_expanded.columns = [col[:-2] for col in categories_expanded.iloc[0,:].tolist()]
    
    # parse category values to convert them to 0 or 1
    categories_clean = pd.DataFrame()
    for col in categories_expanded:
        categories_clean[col] = categories_expanded[col].apply(lambda x: min(abs(int(x[-1])),1))
    
    # recombine cleaned categories data with id #
    categories.reset_index(drop=True, inplace=True)
    categories_clean.reset_index(drop=True, inplace=True)
    categories_clean = pd.concat([categories.id, categories_clean], axis=1)
    
    # merge the messages and categories datasets using the common id and drop unused columns
    # note: none of the samples have label `child_alone`, no additional info, thus drop
    df = messages.merge(categories_clean, on=['id']).drop(['id','original','child_alone'], axis=1)

    # save the clean dataset into an sqlite database
    print(f'Saving data to...\n\tDATABASE: {database_filename}')
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql(database_filename, engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_filename, categories_filename, database_filename = sys.argv[1:]
        ETL(messages_filename, categories_filename, database_filename)
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filenames of the messages and categories datasets as the first and second argument',
              'respectively, as well as the filename of the database to save the cleaned data to as the third argument.',
              '\n\nExample: python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db')

if __name__ == '__main__':
    main()