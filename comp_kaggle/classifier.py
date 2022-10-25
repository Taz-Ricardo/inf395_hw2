#%%
import pandas as pd
from pathlib import Path


download_path = Path.cwd()/'test'

metadata_file = Path.cwd()/'others'/'train.csv'
df = pd.read_csv(metadata_file)
df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
df.head()
#%%

