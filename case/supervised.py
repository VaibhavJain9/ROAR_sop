import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
import statistics as stat
import numpy as np
from sklearn.metrics import confusion_matrix


# Read data
df = pd.read_csv('Total_features.csv')

probe_rate = []

# Define features (x) and target (y)
x = df[['Score', 'arousal_acc_video', 'valence_acc_video', 'GSR_Diff']].values
z = df[['video']].values
y = df[['Probe']].values


f1_score = []
TPR_list = []
FPR_list = []




for i in range(30):
   user_num = i
   print("user number:", user_num)
   array = np.empty(8, dtype=int)
   array.fill(0)

   l = 243*i
   r = l + 243
   x_df = x[l:r]
   y_df = y[l:r]

   z_df = z[l:r]
   x_train, x_test = train_test_split(x_df, test_size=0.20, random_state=42, shuffle=False)
   y_train, y_test = train_test_split(y_df, test_size=0.20, random_state=42, shuffle=False)
   
   z_train, z_test = train_test_split(z_df, test_size=0.20, random_state=42, shuffle=False) 

   result_tuple = np.column_stack((y_train, z_train))
   for yi,zi in result_tuple:
      if yi==1 :
         array[zi-1]+=1
   

   
   probe_rate.append(np.mean(array))

   # Initialize Logistic Regression classifi
   
   logistic_classifier  =LogisticRegression (class_weight= 'balanced', solver = 'newton-cholesky', C =10) 
  
   # Train the classifier on the training data'
   logistic_classifier.fit(x_train, y_train)
  
   # Make predictions on the test data
   probabilities = logistic_classifier.predict_proba(x_test)

# Get the predicted class labels using a threshold of 0.5
   predicted_labels = (probabilities[:, 1] > .3).astype(int)

   y_pred = predicted_labels


   precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
   conf_matrix = confusion_matrix(y_test, y_pred)


   # Extract True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN) from the confusion matrix
   TP = conf_matrix[1, 1]
   FP = conf_matrix[0, 1]
   TN = conf_matrix[0, 0]
   FN = conf_matrix[1, 0]


   # Calculate True Positive Rate (TPR) and False Positive Rate (FPR)
   TPR = TP / (TP + FN)
   FPR = FP / (FP + TN)
   f1_score.append(f1)
   TPR_list.append(TPR)
   FPR_list.append(FPR)


print("Mean F1 Score:", np.mean(f1_score))
print("F1 Scores:", f1_score)
print("Mean TPR:", np.mean(TPR_list))
print("Mean FPR:", np.mean(FPR_list))
print ("std" , stat.stdev (f1_score), stat.stdev(TPR_list), stat.stdev(FPR_list))
print (np.mean(probe_rate))