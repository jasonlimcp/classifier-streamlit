from .utils import data_cleaning, data_engrg

def dataprep(df):
  df = data_cleaning(df)

  df = data_engrg(df)

  print('\nRetrieved', df.shape[0], 'records.')
  print('Pre-processing complete. Following table (first 3 rows) will be ingested by classification script:\n')
  print(df.head(3))

  return df