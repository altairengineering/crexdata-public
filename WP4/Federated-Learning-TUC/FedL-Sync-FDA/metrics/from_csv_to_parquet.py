import glob
import pandas as pd

csv_files = glob.glob('tmp/epoch_metrics/*.csv')

for f in csv_files:
    df = pd.read_csv(f)
    id_name = f.split('/')[-1].split('.')[0]
    df.to_parquet(f'epoch_metrics/{id_name}.parquet')
