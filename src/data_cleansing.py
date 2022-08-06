from .utils import data_cleaning, data_engrg

def dataprep(survive):
  df_survive = survive.copy()

  df_survive = data_cleaning(df_survive)

  df_final = data_engrg(df_survive)

  df = df_final[['Is_Male',
  'Smoke',
  'Diabetes',
  'Age',
  'Ejection Fraction',
  'Sodium',
  'Creatinine',
  'Platelets',
  'Creatine phosphokinase',
  'Blood Pressure',
  'Hemoglobin',
  'BMI',
  'Survive']]

  print('\nRetrieved', df.shape[0], 'patient records.')
  print('Pre-processing complete. Following table (first 3 rows) will be ingested by classification script:\n')
  print(df.head(3))

  return df