import pandas as pd

from datacleanml import DataCleanML

path = '/Users/julius/Library/CloudStorage/GoogleDrive-julius.simonelli@gmail.com/My Drive/JupyterNotebooks/AlgoTrader/data/earn_pg_train_and_val_all_to_20240723.csv'
df = pd.read_csv(path)
cleaner = DataCleanML(config={'detect_binary': True, 'normalize': True, 'datetime_columns': 'Date'}, verbose=True)
clean_df = cleaner.clean(df, save_path='clean_data.csv')
