import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from sklearn.model_selection import train_test_split


import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

columns = ['Model','Train Accuracy','Train Recall','Train Precision','Train F1 Score',
                            'Test Accuracy','Test Recall','Test Precision', 'Test F1 Score']
model_performance = pd.DataFrame(columns=columns)
# model_performance.columns = ['Model','Train Accuracy','Train Recall','Train Precision',
#                             'Test Accuracy','Test Recall','Test Precision']
# Train Test split

def error_matrix(y_train,train_preds,y_test,test_preds,name,dataframe):
    train_accuracy_1= accuracy_score(y_train,train_preds)
    train_recall_1= recall_score(y_train,train_preds)
    train_precision_1= precision_score(y_train,train_preds)
    train_f1 = f1_score(y_train,train_preds)

    test_accuracy_1= accuracy_score(y_test,test_preds)
    test_recall_1= recall_score(y_test,test_preds)
    test_precision_1= precision_score(y_test,test_preds)
    test_f1 = f1_score(y_test,test_preds)
    
    s1=pd.DataFrame([name,train_accuracy_1,train_recall_1,train_precision_1,train_f1,
                                            test_accuracy_1,test_recall_1,test_precision_1,test_f1],
                                           index=model_performance.columns)
    dataframe = pd.concat([dataframe,s1.T], axis=0,ignore_index=True)
    return(dataframe)


def model_building(final):
    name = 'Logistic'
    global model_performance
    X = final.drop(['TAC_Reading'],axis=1)
    y = final['TAC_Reading']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=124)
    # print(X_train.shape)
    # print(X_test.shape)
    # print(y_train.shape)
    # print(y_test.shape)
    scaler = MinMaxScaler()
    X_train[['acc','step_time']] = scaler.fit_transform(X_train[['acc','step_time']])
    X_test[['acc','step_time']]=scaler.transform(X_test[['acc','step_time']])
    X_train.describe()
    
    log = LogisticRegression(class_weight='balanced')

    log.fit(X_train,y_train)
    train_preds = log.predict(X_train)
    train_preds_prob=log.predict_proba(X_train)[:,1]
    test_preds = log.predict(X_test)
    test_preds_prob=log.predict_proba(X_test)[:,1]
    # print('train accuracy:', accuracy_score(y_train,train_preds))
    # print('test accuracy:', accuracy_score(y_test,test_preds))
    ##########################################################
    model_performance=error_matrix(y_train,train_preds,y_test,test_preds,name,model_performance)
    ##########################################################
    fpr, tpr, threshold = roc_curve(y_train, train_preds_prob)
    roc_auc = auc(fpr, tpr)
    roc_df = pd.DataFrame({'FPR':fpr, 'TPR':tpr, 'Threshold':threshold})

    roc_df
    # %matplotlib inline
    # # plt.figure()
    # plt.plot([0,1],[0,1],color='navy', lw=2, linestyle='--')
    # plt.plot(fpr,tpr,color='orange', lw=3, label='ROC curve (area = %0.2f)' % roc_auc)

    # plt.xlabel('FPR')
    # plt.ylabel('TPR')
    # plt.legend(loc="lower right")
    # plt.show()
    roc_df.sort_values('TPR',ascending=False,inplace=True)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]
    optimal_threshold
    custom_threshold=optimal_threshold
    ## To get in 0-1 format vector (pandas Series)
    final_pred_array = pd.Series([0 if x>custom_threshold else 1 for x in train_preds_prob])

    final_test_pred_array = pd.Series([0 if x>custom_threshold else 1 for x in test_preds_prob])
    ## To get True-False format vector (pandas Series)
    final_pred = pd.Series(train_preds_prob > custom_threshold)
    final_test_pred=pd.Series(test_preds_prob > custom_threshold)
    # print('train accuracy:', accuracy_score(y_train,final_pred))
    # print('test accuracy:', accuracy_score(y_test,final_test_pred))

    # print(classification_report(y_train,final_pred))
    model_performance=error_matrix(y_train,final_pred,y_test,final_test_pred,name+' ROC',model_performance)

    # cm1 = confusion_matrix(y_train,final_pred)
    # cm2 = confusion_matrix(y_test,final_test_pred)
    return (model_performance)#,cm1,cm2 