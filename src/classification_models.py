from .utils import scaling, predictor_trainer, f1_score, generate_roc, model_evaluation

def classifications(df):
    import pandas as pd
    from matplotlib import pyplot
    import io
    import time
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from scipy.stats import randint
    from sklearn.model_selection import RandomizedSearchCV
    from imblearn.over_sampling import RandomOverSampler

    x = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    # Standard Scaler
    x_scaled  = scaling(x)

    # Oversampling the minority class
    oversampler = RandomOverSampler(sampling_strategy='minority')
    x_os, y_os = oversampler.fit_resample(x_scaled, y)

    x_train, x_test, y_train, y_test = train_test_split(x_os, y_os, test_size=0.2, random_state = 23, stratify = y_os)

    # Creating tables to hold model evaluation metrics
    df_evaluation = pd.DataFrame(columns=['parameters','f1-score','ROC AUC'])
    model_list = []
    
    # Logistic Regression
    predlabel = 'Logistic Regression'
    model_list.append(predlabel)

    param_dist = {"C": [0.1, 0.5, 1, 5, 10],
                "solver":['lbfgs', 'liblinear']}
    predictor = LogisticRegression()
    predictor_cv = RandomizedSearchCV(predictor, param_dist, cv=5)

    start = time.time()
    predictor, y_pred = predictor_trainer(predictor_cv, x_test, x_train, y_train)
    stop = time.time()
    time_train = stop - start

    print("Tuned {} Parameters: {}, with best CV score of {}.".format(predlabel,predictor.best_params_, predictor.best_score_))
    f1 = f1_score(y_test, y_pred)
    lr_auc = generate_roc(x_test, y_test, predlabel, predictor)
    clf_lr = predictor    
    df_evaluation = pd.concat([df_evaluation, pd.DataFrame.from_records([{'parameters':predictor.best_params_,'f1-score':f1, 'ROC AUC':lr_auc, 'Train Time': time_train}])])
    
    # Random Forest
    predlabel = "Random Forest"
    model_list.append(predlabel)

    param_dist = {"n_estimators": randint(10, 150),
                "max_depth": [None,1,2,3,4,5,6,7]}
    predictor = RandomForestClassifier()
    predictor_cv = RandomizedSearchCV(predictor, param_dist, cv=5)
    
    start = time.time()
    predictor, y_pred = predictor_trainer(predictor_cv, x_test, x_train, y_train)
    stop = time.time()
    time_train = stop - start

    print("Tuned {} Parameters: {}, with best CV score of {}.".format(predlabel,predictor.best_params_, predictor.best_score_))
    f1 = f1_score(y_test, y_pred)
    lr_auc = generate_roc(x_test, y_test, predlabel, predictor)
    clf_rf = predictor
    df_evaluation = pd.concat([df_evaluation, pd.DataFrame.from_records([{'parameters':predictor.best_params_,'f1-score':f1, 'ROC AUC':lr_auc, 'Train Time': time_train}])])
    
    # KNN
    predlabel = 'K-Nearest Neighbours'
    model_list.append(predlabel)

    param_dist = {"n_neighbors": randint(1, 6)}
    predictor = KNeighborsClassifier()
    predictor_cv = RandomizedSearchCV(predictor, param_dist, cv=5)
    
    start = time.time()
    predictor, y_pred = predictor_trainer(predictor_cv, x_test, x_train, y_train)
    stop = time.time()
    time_train = stop - start

    print("Tuned {} Parameters: {}, with best CV score of {}.".format(predlabel,predictor.best_params_, predictor.best_score_))
    f1 = f1_score(y_test, y_pred)
    lr_auc = generate_roc(x_test, y_test, predlabel, predictor)
    clf_nn = predictor
    df_evaluation = pd.concat([df_evaluation, pd.DataFrame.from_records([{'parameters':predictor.best_params_,'f1-score':f1, 'ROC AUC':lr_auc, 'Train Time': time_train}])])
    
    # Naive Bayes
    predlabel = "Naive Bayes"
    model_list.append(predlabel)

    predictor = GaussianNB()
    
    start = time.time()
    predictor, y_pred = predictor_trainer(predictor, x_test, x_train, y_train)
    stop = time.time()
    time_train = stop - start

    f1 = f1_score(y_test, y_pred)
    lr_auc = generate_roc(x_test, y_test, predlabel, predictor)
    clf_nb = predictor
    df_evaluation = pd.concat([df_evaluation, pd.DataFrame.from_records([{'f1-score':f1, 'ROC AUC':lr_auc, 'Train Time': time_train}])])
    
    #Evaluation Metrics
    
    df_final, select_model = model_evaluation(model_list, df_evaluation)

    pyplot.show()

    #Save ROC plot to png (for Web app only)
    roc_plot = io.BytesIO()
    pyplot.savefig(roc_plot, format='png')
    pyplot.clf()
    roc_plot.seek(0)

    return df_final, roc_plot, (clf_lr,clf_rf,clf_nn,clf_nb), select_model