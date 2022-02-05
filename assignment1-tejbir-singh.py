import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import kernel_metrics
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectPercentile, SelectKBest, mutual_info_classif, chi2
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import warnings

### Filter any warning 
warnings.filterwarnings("ignore")

# Reading the CSV file
df = pd.read_csv("clamp_integrated.csv") 
#print('Total Shape :',df.shape)

#Drop the columns with same row values
df=df.drop(['e_cblp','e_cp','e_cparhdr','e_maxalloc','e_sp','e_lfanew'],axis=1)
df=df.drop(['NumberOfSections','CreationYear'],axis=1)

### Check for any non numeric Columns
type_df = pd.DataFrame(df.dtypes).reset_index()
type_df.columns=['cols','type']
#print(type_df[type_df['type']=='object']['cols']) ## Output = 'packer_type'

## Drop the non numeric columns
df = df.drop(['packer_type'],axis=1)
# print(df.columns)

# Getting the X and Y values
X = df.drop(['class'],axis=1)
y=df['class']

# print(X.head())
# print(y)
    
####### Feature Selection ############

# SelectKbest Chi2
SKB_chi2 = SelectKBest(chi2, k=10)
train_data_chi2 = SKB_chi2.fit_transform(X, y)
SKB_cls = SKB_chi2.get_support(indices=True)
SKB_features = df.iloc[:,SKB_cls]

# SelectKBest mutual_info
SKB_mutual = SelectKBest(mutual_info_classif, k=10)
train_data_mt = SKB_mutual.fit_transform(X, y)
SKB_cls = SKB_mutual.get_support(indices=True)
SKB_mt_features = df.iloc[:,SKB_cls]

# SelectPercentile Chi2
SP_chi2 = SelectPercentile(chi2)
train_data_SPchi = SP_chi2.fit_transform(X, y)
SP_cls = SP_chi2.get_support(indices=True)
SP_features = df.iloc[:,SP_cls]

# SelectPercentile Mutual_info
SP_mutual = SelectPercentile(mutual_info_classif)
train_data_SPmt = SP_mutual.fit_transform(X, y)
SP_mt_cls = SP_mutual.get_support(indices=True)
SP_mt_features = df.iloc[:,SP_mt_cls]


####### Selecting the best features ###########

# cases = [SKB_chi2,SKB_mutual,SP_chi2,SP_mutual]
# i=0
# for case in cases:
#     class_pipeline = pipeline.make_pipeline(case,SVC(kernel='linear'))
#     scores = cross_val_score(class_pipeline, X, y, cv=10,scoring="f1")
#     print("Scores for ",'\033[1m'+ method[i]+'\033[0m')
#     print("Cross Validation accuracy scores: ", scores)
#     print('\nCross Validation accuracy: %.3f +/- %.3f\n' % (np.mean(scores),np.std(scores)))
#     i+=1

# ** We will use SelectKBest Chi2 **

features_selected = SKB_features
# print(features_selected)

# Printing the correlation matrix for features selected

matrix = df.loc[:,features_selected.columns].corr()
plot = sns.heatmap(matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag')
plot.set(title = "Correlation matrix of SelectKbest Chi2 feautures\n")
plt.show()

########## Standardizing the data
scaler = StandardScaler()
X_new = scaler.fit_transform(features_selected)

# Train test split
x_train, x_test, y_train, y_test = train_test_split(X_new,y,test_size=0.3)

############# Model Training #############
models = []
predict_data = []
acc_score = []

### Random Forest Classifier model 

rf = RandomForestClassifier(criterion='gini',random_state=1,n_estimators=20) 
rf_trained = rf.fit(x_train,y_train)
models.append(rf_trained)

### SVC model 
svc = SVC(random_state=1, kernel='linear')
svc_trained = svc.fit(x_train,y_train)
models.append(svc_trained)

### K nearest neighbour
knn = KNeighborsClassifier()
knn_trained = knn.fit(x_train, y_train)
models.append(knn_trained)

###### Cross validation

# print(models)
# print(trained_data)

# ### Holdout method
i=0
title = ['RF','SVC','KNN']
for model in models:
    
    predict = model.predict(x_test)
    matrix = confusion_matrix(y_test,predict)
    class_report = classification_report(y_test,predict)
    acc_report = accuracy_score(y_test,predict)
    acc_score.append(acc_report)
    predict_data.append(predict)
    print("\n***************",model,"***************")
    print(class_report)
    print("Accuracy score using Holdout method: ",acc_report)
    print("confusion matrix: ",matrix)

    plot = sns.heatmap(matrix, annot=True, cmap='vlag')
    plot.set(title = "Correlation for "+title[i]+" with holdout method\n")
    plt.show()
    i+=1

### K-fold method
i=0
for model in models:
    
    kfold = model_selection.KFold(shuffle=True)
    # cross_val = cross_val_score(model, x_train, y_train, cv=kfold, scoring='f1')
    predict = cross_val_predict(model,x_test,y_test,cv=kfold)
    class_report = classification_report(y_test,predict)
    acc_report = accuracy_score(y_test,predict)
    matrix = confusion_matrix(y_test,predict)
    acc_score.append(acc_report)
    predict_data.append(predict)
    print("\n***************",model,"***************")
    print(class_report)
    print("Accuracy score using K-fold method: ",acc_report)
    print("confusion matrix: ",matrix)
    
    plot = sns.heatmap(matrix, annot=True, cmap='vlag')
    plot.set(title = "Correlation for "+title[i]+" with K-fold method\n")
    plt.show()
    i+=1


### Leave one out method
i=0
for model in models:
    
    leave_oo = model_selection.LeaveOneOut()
    # cross_val = cross_val_score(model, x_train, y_train, cv=leave_oo, scoring='f1')
    predict = cross_val_predict(model,x_test,y_test,cv=leave_oo)
    class_report = classification_report(y_test,predict)
    acc_report = accuracy_score(y_test,predict)
    matrix = confusion_matrix(y_test,predict)
    acc_score.append(acc_report)
    predict_data.append(predict)
    print("\n***************",model,"***************")
    print(class_report)
    print("Accuracy score using Leave one out method: ",acc_report)
    print("Confusion Matrix: ",matrix)

    plot = sns.heatmap(matrix, annot=True, cmap='vlag')
    plot.set(title = "Correlation for "+title[i]+" with leave-one out method\n")
    plt.show()
    i+=1

## Loop over the predicted data and create ROC curve
x_axis = ['H-RF','H-SVC','H-KNN','KF-RF','KF-SVC','KF-KNN','L-RF','L-SVC','L-KNN']
i=0
for data in predict_data:
    fpr, tpr, thresholds = roc_curve(y_test, data)
    auc = roc_auc_score(y_test,data)
    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (x_axis[i], auc))
    i+=1
## plotting the ROC curve 

plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show() 


### bar graph to compare the accuracy of various models

### creating the bar plot
## Defining the colours just for fun :)
c = ['#B22027','#B22027','#B22027','#20B2AA','#20B2AA','#20B2AA','#008000','#008000','#008000' ]
plt.bar(x_axis, acc_score, color =c)
plt.xlabel("Models Used")
plt.ylabel("Accuracy Score")
plt.title("Models VS Accuracy Score")
plt.show()

####################################################################################################

model =[RandomForestClassifier(),SVC(),KNeighborsClassifier()]
grid_predict =[]
############ Using Grid Search CV 
parameters1 = {
'criterion': ['gini','entropy'],
'max_features': [10],
}

parameters3 = {
'weights': ['uniform', 'distance'],
'n_neighbors': [5]
}

parameters2 = {
'kernel': ['linear','sigmoid','rbf'],
'random_state':[1]}
pars = [parameters1, parameters2, parameters3]
for i in range(len(pars)):
    gs = GridSearchCV(model[i], pars[i], refit=True, n_jobs=-1)
    gs = gs.fit(x_train, y_train)
    # print ("Best Score for ",model[i])
    # print (gs.best_score_)
    # params = gs.best_params_
    # print(params)
    # pred = gs.set_params(params=params.keys)
    pred = gs.best_estimator_.predict(x_test)
    grid_predict.append(pred)
    print(confusion_matrix(y_test,pred))
    print(classification_report(y_test,pred))
    
x_axis = ['RF','SVC','KNeighbour']
i=0
for data in grid_predict:
    fpr, tpr, thresholds = roc_curve(y_test, data)
    auc = roc_auc_score(y_test,data)
    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (x_axis[i], auc))
    i+=1
## plotting the ROC curve 

plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show() 

# ############################################################
# ### END OF CODE ###