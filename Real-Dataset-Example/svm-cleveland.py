import pandas as pd
import matplotlib.pyplot as plt

#sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('Cleveland Dataset.csv')

X = data.drop(['target'] ,axis="columns")
y = data['target']

estimator = SVC(kernel="linear")
selector = RFE(estimator, n_features_to_select=6, step=1)
selector = selector.fit(X, y)

# classical SVM
X = data[['ca','cp','thal','exang','slope']]
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Create svm Classifier
ClassifierSVM = SVC()

# Train the model using the training set
ClassifierSVM.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = ClassifierSVM.predict(X_test)

# Model Accuracy: 
print("SVM classification test score:",metrics.accuracy_score(y_test, y_pred))

# classification report of SVM
expected_y  = y_test
predicted_y = ClassifierSVM.predict(X_test) 

# print classification report and confusion matrix for svm classifier
print("Classification report: \n", metrics.classification_report(expected_y, predicted_y))
print("Confusion matrix: \n", metrics.confusion_matrix(expected_y, predicted_y))