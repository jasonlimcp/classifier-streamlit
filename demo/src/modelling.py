import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from matplotlib import pyplot
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
import io

def data_preparation(df):
    df = df.replace({'work_type': {'Never_worked':'children'}})
    df = df.loc[df['gender']!= 'Other']
    df = df.replace({'gender': {'Male': 1, 'Female': 0}, 'Residence_type': {'Urban': 1, 'Rural': 0},'ever_married': {'Yes': 1, 'No': 0}})
    df = df.drop(['id'],axis=1)
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    df_target = df[['stroke']]
    df = df.drop(['stroke'], axis=1)
    df = pd.get_dummies(df, columns = ['work_type','smoking_status'])
    df = df.join(df_target)

    return df

from imblearn.over_sampling import RandomOverSampler
from scipy.stats import randint

def svc_model(df):
    x = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    ss = StandardScaler() 
    x_scaled = ss.fit_transform(x)

   
    oversample = RandomOverSampler(sampling_strategy='minority')
    X_over, y_over = oversample.fit_resample(x, y)

    x_train, x_test, y_train, y_test = train_test_split(X_over,y_over,test_size=0.2,random_state=11,stratify=y_over)

    param_dist = {"n_estimators": randint(10, 150),
                "max_depth": [None,1,2,3,4,5,6,7]}

    classifier = RandomForestClassifier()
    classifier_cv = RandomizedSearchCV(classifier, param_dist, cv=5)
    classifier_cv.fit(x_train, y_train)
    y_pred = classifier_cv.predict(x_test)

    print("Tuned SVC Parameters: {}, with best CV score of {}.".format(classifier_cv.best_params_, classifier_cv.best_score_))

    eval_dict = classification_report(y_test, y_pred,output_dict=True)
    f1 = eval_dict.get('weighted avg').get('f1-score')

    ns_probs = [0 for _ in range(len(y_test))]
    lr_probs = classifier_cv.predict_proba(x_test)
    lr_probs = lr_probs[:, 1]
    lr_auc = roc_auc_score(y_test, lr_probs)

    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--')
    pyplot.plot(lr_fpr, lr_tpr, marker='.', label='%s (AUC = %.3f)' %('SVC', lr_auc))
    pyplot.xlabel('False Positive Rate (FPR)')
    pyplot.ylabel('True Positive Rate (TPR)')
    pyplot.legend(loc="lower right")
    pyplot.title("ROC Curves for Classifiers")
    pyplot.show()

    roc_plot = io.BytesIO()
    pyplot.savefig(roc_plot, format='png')
    pyplot.clf()
    roc_plot.seek(0)

    return roc_plot