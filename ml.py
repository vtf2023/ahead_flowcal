# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 15:24:33 2023

@author: kuowei huang

"""
import seaborn
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, validation_curve, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, make_scorer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest 



def calculateCrossValScore(models, X_train, y_train): 
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    scorer_acc = make_scorer(accuracy_score)
    scorer_rec = make_scorer(recall_score)
    ret_acc = []
    ret_rec = []

    if isinstance(models, dict):
        for model in models.keys():
            scores_acc = cross_val_score(estimator=models[model],
                                     X=X_train,
                                     y=y_train,
                                     cv=kf, scoring = scorer_acc,   
                                     n_jobs=1)
            scores_recall = cross_val_score(estimator=models[model],
                                     X=X_train,
                                     y=y_train,
                                     cv=kf, scoring = scorer_rec,   
                                     n_jobs=1)
            
            print('CV accuracy of ' + model + ': %.3f +/- %.3f' % (np.mean(scores_acc), np.std(scores_acc)))
            print('CV recall of ' + model + ': %.3f +/- %.3f' % (np.mean(scores_recall), np.std(scores_recall)))
            ret_acc.append('CV accuracy of ' + model + ': %.3f +/- %.3f' % (np.mean(scores_acc), np.std(scores_acc)))
            ret_rec.append('CV recall of ' + model + ': %.3f +/- %.3f' % (np.mean(scores_recall), np.std(scores_recall)))
    else:
        scores_acc = cross_val_score(estimator=models,
                                 X=X_train,
                                 y=y_train,
                                 cv=kf, scoring = scorer_acc,   
                                 n_jobs=1)
        scores_recall = cross_val_score(estimator=models,
                                 X=X_train,
                                 y=y_train,
                                 cv=kf, scoring = scorer_rec,   
                                 n_jobs=1)
        
        print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores_acc), np.std(scores_acc)))
        print('CV recall: %.3f +/- %.3f' % (np.mean(scores_recall), np.std(scores_recall)))
        ret_acc.append('CV accuracy: %.3f +/- %.3f' % (np.mean(scores_acc), np.std(scores_acc)))
        ret_rec.append('CV recall: %.3f +/- %.3f' % (np.mean(scores_recall), np.std(scores_recall)))
    
    return ret_acc, ret_rec



def plotValidationCurve(train_scores, vali_scores, x_axis, x_label, file_name, y_b = 0):
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(vali_scores, axis=1)
    test_std = np.std(vali_scores, axis=1)
    
    plt.plot(x_axis, train_mean, 
             color='blue', marker='o', 
             markersize=5, label='training accuracy')
    
    plt.fill_between(x_axis, train_mean + train_std,
                     train_mean - train_std, alpha=0.15,
                     color='blue')
    
    plt.plot(x_axis, test_mean, 
             color='green', linestyle='--', 
             marker='s', markersize=5, 
             label='validation accuracy')
    
    plt.fill_between(x_axis, 
                     test_mean + test_std,
                     test_mean - test_std, 
                     alpha=0.15, color='green')
    
    plt.grid()
    
    plt.legend(loc='lower right')
    plt.xlabel(x_label)
    plt.ylabel('Accuracy')
    plt.ylim([y_b, 1.0])
    plt.tight_layout()
    plt.savefig(file_name, dpi=300)
    plt.show()



def main():
    
    np.random.seed(10)
    
    X_heal = np.load('healthy_g1.npy', allow_pickle=True)
    X_sick = np.load('sick_g1.npy', allow_pickle=True)
    
    
    X = np.zeros((len(X_heal)+len(X_sick), 62))
    y = np.concatenate((np.zeros(len(X_heal)), np.ones(len(X_sick))), axis =0)
    
    # creating new features by averaging and std. for demo only
    # better feature engineering should be used.  
    for i in range(len(X_heal)):
        for j in range(31):
            X[i, j] = np.mean(X_heal[i][:,j])
                 
    for i in range(len(X_heal)):
        for j in range(31):
            X[i, j+31] = np.std(X_heal[i][:,j])
        
    for i in range(len(X_sick)):
        for j in range(31):
            X[len(X_heal)+i, j] = np.mean(X_sick[i][:,j])
                   
    for i in range(len(X_sick)):
        for j in range(31):
            X[len(X_heal)+i, j+31] = np.std(X_sick[i][:,j])
           
    ### some QC steps might be needed here ###
    
    
    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 23)
    
    # normalizaion
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    
    ### feature selection might be applied here if wanted.
    
    
    # init models
    model_svc = SVC(class_weight = 'balanced',random_state = 23)
    model_nn = MLPClassifier(random_state=23)
    model_rf = RandomForestClassifier(criterion = "gini",class_weight = 'balanced', random_state=23)
    model_adaboost = AdaBoostClassifier(estimator=DecisionTreeClassifier(class_weight = 'balanced'),random_state = 23)
    
    classifiers = {"SVM" :model_svc,  
                    "NN": model_nn,
                    "RF": model_rf,
                    "AdaBoost": model_adaboost}
        
    classifiers1 = copy.deepcopy(classifiers)
    score_acc_1, score_recall_1 = calculateCrossValScore(classifiers1, X_train_scaled, y_train)
    
    
    
    # RF tuning
    rf_n_estimators = range(10,150,10)
    train_scores, vali_scores = validation_curve(
                                estimator=RandomForestClassifier(criterion = "gini",class_weight = 'balanced',random_state = 53), 
                                X=X_train_scaled, 
                                y=y_train,
                                param_name='n_estimators', 
                                param_range=rf_n_estimators )
    
    plotValidationCurve(train_scores, vali_scores, rf_n_estimators, \
                        "rf_n_estimators", "rf_1.png",y_b = 0.3)
    # n_estimators = 100 
    
    
    # find feature importance
    marker_channel_map = pd.read_csv("EU_marker_channel_mapping.csv")
    channels_names = marker_channel_map.loc[marker_channel_map['use']==1]["PxN(channel)"].tolist()
    
    feat_labels =  []
    
    for i in range(X_train.shape[1]):
        if i < 31:
            feat_labels.append(channels_names[i]+ "-ave") 
        else:
            feat_labels.append(channels_names[i-31]+ "-std")
    
    forest = RandomForestClassifier(n_estimators=100,random_state=1)
    
    forest.fit(X_train_scaled, y_train)
    importances = forest.feature_importances_
    
    indices = np.argsort(importances)[::-1]
    
    for f in range(X_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
    
    
    # SVM tuning
    svc_params_grid = {'kernel': ['rbf', 'sigmoid', 'poly', 'linear'], 
                       'gamma': ["auto","scale", 0.0001, 0.001, 0.01, 0.1]}
    gs_svc_1 = GridSearchCV(SVC(random_state = 52), svc_params_grid, cv=5)
    gs_svc_1.fit(X_train_scaled, y_train)
    print('Best score for training data:', gs_svc_1.best_score_,"\n") #  0.747
    print('Best Kernel:',gs_svc_1.best_estimator_.kernel,"\n") #poly
    print('Best Gamma:',gs_svc_1.best_estimator_.gamma,"\n") #auto
    
    
    
    ## select 10 features. Can be dimension reduction
    selected_features_10 = SelectKBest(k=10).fit(X_train_scaled, y_train)
    selected_features_10.get_feature_names_out()
    X_train_selected_10 = selected_features_10.transform(X_train_scaled)
    X_test_selected_10 = selected_features_10.transform(X_test_scaled)
    
    gs_svc_5 = GridSearchCV(SVC(random_state = 52), svc_params_grid, cv=10)
    gs_svc_5.fit(X_train_selected_10, y_train)
    print('Best score for training data:', gs_svc_5.best_score_,"\n") #0.71666
    print('Best Kernel:',gs_svc_5.best_estimator_.kernel,"\n") #sigmoid
    print('Best Gamma:',gs_svc_5.best_estimator_.gamma,"\n") #auto
    
    
    
    #NN tuning
    params_grid_nn = {'hidden_layer_sizes': [ (60, 10), (60, 30), (60,), (100,)] , 
                     'activation': ['relu', 'logistic', 'tanh'], 'alpha': [0.0001,0.001,0.01, 0.05],
                   'learning_rate': ['constant','adaptive']}
    
    gs_nn = GridSearchCV(estimator=MLPClassifier(random_state=1, max_iter=300,solver='adam', tol=0.001),
                         param_grid=params_grid_nn,
                         scoring='accuracy',
                         cv=5, n_jobs=1)
    
    gs_nn = gs_nn.fit(X_train_scaled, y_train)
    print(gs_nn.best_score_) #0.808333
    print(gs_nn.best_params_)
    ##{'activation': 'logistic', 'alpha': 0.05, 'hidden_layer_sizes': (100,), 'learning_rate': 'constant'}
    
    
    
    # AdaBoost
    ada_n_estimators = range(10, 200, 10)
    train_scores, vali_scores = validation_curve(
                                estimator=AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2) ,random_state = 53 ), 
                                X=X_train_scaled, 
                                y=y_train, 
                                param_name='n_estimators', 
                                param_range=ada_n_estimators,
                                cv=5)
    
    plotValidationCurve(train_scores, vali_scores, ada_n_estimators, "n_estimator", "valiCurve_ada_1.png",y_b = 0.5)
    ##  n = 100
    
    
    model_svc = SVC(kernel='poly',gamma='auto', random_state = 23)
    model_svc_2 = SVC(kernel='sigmoid',gamma='auto', random_state = 23)
    model_nn = MLPClassifier(hidden_layer_sizes=(100,), activation= 'logistic', alpha= 0.05,random_state=23, learning_rate='constant')
    model_rf = RandomForestClassifier(criterion = "gini", n_estimators=100, random_state=23,class_weight = 'balanced')
    model_adaboost = AdaBoostClassifier(estimator=DecisionTreeClassifier(), n_estimators = 100,random_state = 23)
    
    classifiers_2 ={}
    classifiers_2 = {"SVM" :model_svc,
                     "SVM_2":model_svc_2,
                    "NN": model_nn, 
                    "RF": model_rf,
                    "AdaBoost": model_adaboost}
    
    
    score_acc_3, score_recall_3 = calculateCrossValScore(classifiers_2, X_train_scaled, y_train)
    
    
    # testing set 
    for key in classifiers_2.keys():
        if key == "SVM_2":
            classifiers_2[key].fit(X_train_scaled, y_train)
            y_pred = classifiers_2[key].predict(X_test_scaled)
            test_acc = accuracy_score(y_test, y_pred)
            test_recall = recall_score(y_test, y_pred)
            print("Test acc of " + key +": =" + str(test_acc))
            print("Test recall of " + key +": =" +str(test_recall))
        else:    
            classifiers_2[key].fit(X_train_scaled, y_train)
            y_pred = classifiers_2[key].predict(X_test_scaled)
            test_acc = accuracy_score(y_test, y_pred)
            test_recall = recall_score(y_test, y_pred)
            print("Test acc of " + key +": =" + str(test_acc))
            print("Test recall of " + key +": =" +str(test_recall))



if __name__ == "__main__":
    main()
