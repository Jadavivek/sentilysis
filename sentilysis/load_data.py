import pandas as pd 
import os

def load_combined_data() -> tuple:
    # load all the csv files from the data directory
    data_dir = 'data'
    files = os.listdir(data_dir)
    dataframes = []
    for file in files:
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(data_dir, file))
            dataframes.append(df)

    # concatenate all the dataframes into one
    df = pd.concat(dataframes, ignore_index=True)

    # drop the columns that are not needed
    X = df['text']
    y = df['label']

    # save the dataframe to a csv file
    df.to_csv('data/combined.csv', index=False)

    return X, y

load_combined_data()