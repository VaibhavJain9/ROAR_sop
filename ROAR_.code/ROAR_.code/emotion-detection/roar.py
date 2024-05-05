import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from modAL.models import ActiveLearner
from sklearn.metrics import classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
import numpy as np
import random

df = pd.read_csv('data.csv')

# Assume the last column is the target variable
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
# y_pred_total stores all the predictions combined
df1 = pd.read_csv('user_data.csv', delimiter=';')

# Extract "start" and "End" columns , start and end are index for each user entries
user_data = df1[['start', 'End']].values 

f1_scores = []
total_count =0 


for i in range (32):
 user_num = i

 print ("user number: " , user_num)
 first_data_point_x = x[user_data[user_num][0]].reshape(1, -1)
 first_data_point_y = np.array([y[user_data[user_num][0]]])

 # Initialize the active learner
 learner = ActiveLearner(
     estimator=RandomForestClassifier(),
 )
 learner.fit(first_data_point_x, first_data_point_y)
 
 x_train, x_test = train_test_split (x[user_data[user_num][0]:user_data[user_num][1]],test_size=0.20, random_state=42, shuffle = False)
 y_train,y_test = train_test_split (y[user_data[user_num][0]:user_data[user_num][1]],test_size=0.20, random_state=42, shuffle =False)
 # Road parameters
 epsilon =0.1
 learning_rate = 0.01
 threshold = 0.7
 positive_reward = 1.0
 negative_reward = -2.0
 probe_count = 1

 for idx, (data_point, true_label) in enumerate(zip(x_train, y_train)):
     # Make predictions
     prediction_probabilities = learner.predict_proba(data_point.reshape(1, -1))

     
     predicted_class = learner.predict(data_point.reshape(1, -1))[0]

    
     max_probability = np.max(prediction_probabilities)
    
     rand = random.uniform(0, 1)
     if rand <= epsilon:
        
         learner.teach(data_point.reshape(1, -1), np.array([true_label]))  # Wrap true_label in an array
         probe_count += 1
 

     else:
         # Check uncertainty and query for label if uncertain
         if max_probability <= threshold:
               # Teach the model with the new labeled data
             learner.teach(data_point.reshape(1, -1), np.array([true_label]))  # Wrap true_label in an array
             probe_count += 1
             
                     # Calculate reward and update policy
             if predicted_class == true_label:
                 reward = negative_reward
             else:
                 reward = positive_reward
             
             threshold = min(threshold * (1 + learning_rate * (1 - pow(2, reward / negative_reward))), 1.0)
             
  
     
 y_pred = learner.predict(x_test)

 print ('threshold', threshold)

 precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
 # Print the results
 f1_scores.append(f1)
 print (f1)

 total_count += probe_count
 print( " probe count :", probe_count, "number of samples :" , user_data[user_num][1] -user_data[user_num][0] , )
 
print (total_count)
print(f1_scores)
print (np.mean(f1_scores))