from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
import pandas as pd

def data_cleaning(df):
    '''Performs data cleansing on fields within the original 'survive table, as explored in EDA'''
    df = df.replace({'Ejection Fraction': {'L': 'Low', 'N': 'Normal'}})
    df['Smoke'] = df['Smoke'].str.lower()
    df = df.replace({'Survive': {'Yes': True, 'No': False, '1': True,'0': False}, 'Smoke': {'yes': True, 'no': False},'Gender': {'Male': True, 'Female': False}})
    df = df.rename(columns={'Gender': 'Is_Male'})
    df = df.replace({'Ejection Fraction': {'Low': 1, 'Normal': 2, 'High': 3}, 'Diabetes': {'Normal': 0, 'Pre-diabetes': 1,'Diabetes': 2}})
    df['Age'] = df['Age'].abs()

    return df

def data_engrg(df):
    '''Create BMI field, and impute missing values where required.'''
    df['BMI'] = df.apply(lambda x: x['Weight']/((x['Height']/100)**2), axis=1)
    df = df.drop(['Height','Weight'],axis=1)
    df = df.drop(['ID'],axis=1)

    df_final = df.copy()
    df_final['Creatinine'] = df_final['Creatinine'].fillna(df_final['Creatinine'].median())

    df_final = df_final.drop(['Favorite color'],axis=1)

    return df_final

def scaling(x):
    '''Uses Sklearn Standard Scaler to normalize input variables'''
    ss = StandardScaler() 
    x_scaled = ss.fit_transform(x)

    return x_scaled

def predictor_trainer(predictor, x_test, x_train, y_train):
    '''Fits chosen classifier and runs prediction'''
    predictor.fit(x_train, y_train)
    y_pred = predictor.predict(x_test)

    return predictor, y_pred

def f1_score(y_test, y_pred):
    '''Calculate F1 score from precision and recall metrics in classification report from predicted labels'''
    eval_dict = classification_report(y_test, y_pred,output_dict=True)
    f1 = eval_dict.get('weighted avg').get('f1-score')

    return f1

def generate_roc(x_test, y_test, predlabel, predictor):
    '''Generate ROC metrics and Curve'''
    ns_probs = [0 for _ in range(len(y_test))]
    lr_probs = predictor.predict_proba(x_test)
    lr_probs = lr_probs[:, 1]
    lr_auc = roc_auc_score(y_test, lr_probs)

    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--')
    pyplot.plot(lr_fpr, lr_tpr, marker='.', label='%s (AUC = %.3f)' %(predlabel, lr_auc))
    pyplot.xlabel('False Positive Rate (FPR)')
    pyplot.ylabel('True Positive Rate (TPR)')
    pyplot.legend(loc="lower right")
    pyplot.title("ROC Curves for Classifiers")
    pyplot.show(block = False)

    return lr_auc

def model_evaluation(model_list, df):
    '''Based on passed metrics, generate Evaluation metrics table. Metrics are not calaculated in this table.'''
    print('\n\033[1m' + 'Evaluation Metrics for Binary Classifier Models' + '\033[0m')
    
    indexnames = model_list
    df = df.set_index([indexnames])

    df['Classifier'] = df.index
    df[['f1-score','ROC AUC']] = df[['f1-score','ROC AUC']].apply(pd.to_numeric)

    df['parameters'] = df['parameters'].fillna(0)
    df['parameters'] = df['parameters'].apply(lambda x: x.items() if x != 0 else '')

    df['Score'] = df.apply(lambda x: 1/(x['f1-score']*x['ROC AUC']), axis = 1)
    df = df.sort_values(by=['Score','Train Time'], ascending = True)

    print(df[['parameters','f1-score','ROC AUC','Train Time']].round(3))

    select_model = df['Classifier'].iloc[0]
    print('\nSelected Binary Classifier:', '\033[1m' + df['Classifier'].iloc[0] + '\033[0m\n')

    return df, select_model