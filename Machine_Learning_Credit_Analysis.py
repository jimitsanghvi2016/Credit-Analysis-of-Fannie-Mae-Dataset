
# coding: utf-8


# Importing all necessary libraries
# Please note this is just a sample analysis on a sample dataset comprising of 12204567 loans
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib as plt
import seaborn as sns



# Reading the data file in csv
df = pd.read_csv("data_final.csv")

df['Default'].value_counts()

df.head()



# Deleting all rows with na values
df = df.dropna(axis=1, how='any')



# The function getDummies will get create Dummy Variables for all Categorical Variables
def getDummies(df):
    columns = df.columns[df.isnull().any()]
    nan_cols = df[columns]

    df.drop(nan_cols.columns, axis=1, inplace=True)

    cat = df.select_dtypes(include=['object'])
    num = df.drop(cat.columns, axis=1)

    data = pd.DataFrame()
    for i in cat.columns:
        tmp = pd.get_dummies(cat[i], drop_first=True)
        data = pd.concat([data, tmp], axis=1)

    df = pd.concat([num,data,nan_cols], axis=1).reset_index(drop=True)
    return df


# Creating the dataset with dummy variables	
df = getdummies(df)

# The loan dataset is unbalanced i.e. 99% of the loans are categorized as non-default and only 1% of them are default. 
# So to balance the dataset we use SMOTE (Synthetic Minority Oversampling Technique.
# It Over Samples the data with the minority class using KNN (K Nearest Neighbor)
sm = imblearn.over_sampling.SMOTE(sampling_strategy='auto', random_state=None, k_neighbors=5, m_neighbors='deprecated', out_step='deprecated', kind='deprecated', svm_estimator='deprecated', n_jobs=1, ratio=None)

# X has all the independent variables and y has the dependent variable i.e. Default
y = df['Default'].values
X = df.drop(['Default'], axis=1).values

# Applying SMOTE on the Dataset
X_res, y_res = sm.fit_resample(X, y)


# Splitting the data into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size = 0.25, random_state=0)



# Random Forest
model = RandomForestClassifier(n_estimators=200)
model = model.fit(X_train, y_train)

# Predicting on the testing set
predict = model.predict(X_test)

# Getting the precision, recall, and f-score of the model i.e. model accuracy
print(classification_report(y_test, predict))

# Visualizing the confusion matrix
cm = metrics.confusion_matrix(y_test, predict).T
cm = cm.astype('float')/cm.sum(axis=0)

fig, ax = plt.pyplot.subplots()
sns.heatmap(cm, annot=True, cmap='Blues');
ax.set_xlabel('True Label')
ax.set_ylabel('Predicted Label')
ax.xaxis.set_label_position('top')

# Visualizing the ROC AUC Curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict_proba(X_test)[:,1])
roc_auc = metrics.roc_auc_score(y_test, predict)

plt.pyplot.plot(fpr, tpr, lw=1, label='AUC = %0.2f'%(roc_auc))
plt.pyplot.plot([0, 1], [0, 1], '--k', lw=1)
plt.pyplot.xlabel('False Positive Rate')
plt.pyplot.ylabel('True Positive Rate')
plt.pyplot.title('Random Forest ROC')
plt.pyplot.legend(loc="lower right", frameon = True).get_frame().set_edgecolor('black')



# Logistic Regression
logisticRegr = LogisticRegression()

logisticRegr = logisticRegr.fit(X_train, y_train)

predict1 = logisticRegr.predict(X_test)

print(classification_report(y_test, predict1))

cm = confusion_matrix(y_test, predict1).T
cm = cm.astype('float')/cm.sum(axis=0)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, cmap='Blues');
ax.set_xlabel('True Label')
ax.set_ylabel('Predicted Label')
ax.xaxis.set_label_position('top')


# SVM
clf = SVC(gamma='auto', probability=True)

clf = clf.fit(X_train, y_train)


predict3 = clf.predict(X_test)
print(classification_report(y_test, predict2))


# Visualizing the ROC Curve of all three Models
fpr, tpr, thresholds = roc_curve(y_test, logisticRegr.predict_proba(X_test)[:,1])
roc_auc = roc_auc_score(y_test, predict1)

fpr1, tpr1, thresholds1 = roc_curve(y_test, model.predict_proba(X_test)[:,1])
roc_auc1 = roc_auc_score(y_test, predict)


fpr2, tpr2, thresholds2 = roc_curve(y_test, clf.predict_proba(X_test)[:,1])
roc_auc2 = roc_auc_score(y_test, predict3)

plt.plot(fpr, tpr, lw=1, label='Logistic Regression = %0.2f'%(roc_auc))
plt.plot(fpr1, tpr1, lw=1, label='Random Forest = %0.2f'%(roc_auc1))
plt.plot(fpr2, tpr2, lw=2, label='SVM = %0.2f'%(roc_auc2))


plt.plot([0, 1], [0, 1], '--k', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right", frameon = True).get_frame().set_edgecolor('black')

