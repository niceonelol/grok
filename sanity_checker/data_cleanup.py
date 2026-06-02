import os
import re
import pandas as pd

"""
folder_path = 'sanity_checker/phdim_data'

for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        #df['phdim'] = df['phdim'].str.extract(
        
        df = df.rename(columns={
            'test_accuracy': 'val_accuracy',
            'phdim': 'phdim_0'
        })
        
        df.to_csv(file_path, index=False)
"""
