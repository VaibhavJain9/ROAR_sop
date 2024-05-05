import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
from modAL.models import ActiveLearner



import numpy as np
import random
# import sys
# # setting output stream as a file 
# with open('file1.txt', 'w') as sys.stdout:
df = pd.read_csv('data.csv')

# Assume the last column is the target variable
x = df.iloc[:, :-1].values


y = df.iloc[:, -1].values
# y_pred_total stores all the predictions combined

f1_score =[]

# Load the CSV file into a DataFrame with semicolon (;) as the delimiter
df1 = pd.read_csv('user_data.csv', delimiter=';')

# Extract "start" and "End" columns , start and end are index for each user entries
user_data = df1[['start', 'End']].values 



for i in range (32):
 user_num = i
 print ("user number: " , user_num)
 
 
 x_train, x_test = train_test_split (x[user_data[user_num][0]:user_data[user_num][1]],test_size=0.20, random_state=42, shuffle = False)
 y_train,y_test = train_test_split (y[user_data[user_num][0]:user_data[user_num][1]],test_size=0.20, random_state=42,shuffle = False)
 rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

 

 # Train the classifier on the training data
#  rf_classifier.fit(x_train, y_train)
 
#  # Make predictions on the test data
#  y_pred = rf_classifier.predict(x_test)
 learner =RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Initialize the ActiveLearner
 learner.fit(x_train, y_train)
    
    # Make predictions
 y_pred = learner.predict(x_test)
 

 precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
 f1_score.append(f1)
#  importances = learner.feature_importances_

# # Print feature importances
#  for i, importance in enumerate(importances):
#     print(f"Feature {i+1}: Importance = {importance}")

print (np.mean(f1_score))
print (f1_score)
user_labels = [f"{i + 1}" for i in range(len(f1_score))]
plt.bar(user_labels, f1_score, color='blue')
plt.xlabel('Users')
plt.ylabel('F1 Score')
plt.title('F1 Score for Each User')
#plt.show()