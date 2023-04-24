import pandas as pd

data = pd.read_csv('C:/Users/Playdata/Desktop/찌르레기/gitproject/datas/train.csv')

data.isna().sum()
