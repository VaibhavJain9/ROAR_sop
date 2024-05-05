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
# Create the map
m = {
    -2: 3,
    0: 0,
    1: 1,
    2: 2
}
# Extract "start" and "End" columns , start and end are index for each user entries
user_data = df1[['start', 'End']].values 

f1_scores = []
total_count =0 

freq_class = []

for i in range(32):
    start = user_data[i][0]
    end = user_data[i][1]
    freq_map = {}
    max_freq_value = None
    max_freq = -1
   

    for j in range(int(start), min(int(end) + 1, len(y))):

        

        if y[j] in freq_map:
            freq_map[y[j]] += 1
        else:
            freq_map[y[j]] = 1

        if freq_map[y[j]] > max_freq:
            max_freq = freq_map[y[j]]
            max_freq_value = y[j]

    freq_class.append(max_freq_value)


vec = np.array([
    [0],[0],[0],[0]
  ])

for i in range (32):
 user_num = i
 initial_state = freq_class[i]
 idx = m[initial_state]
 vec = np.array([
    [0],[0],[0],[0]
  ])
 vec[idx] =1 # change 

 freq = np.array([
    [0],[0],[0],[0] ])
 freq[idx] =1 # change 
 
 trans_freq = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
  ])
 trans_freq[idx][idx] =1 #change 
 trans_matrix = np.array([
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0]
])
 #trans_matrix[idx][idx] = 1

 print ("user number: " , user_num)
 first_data_point_x = x[user_data[user_num][0]].reshape(1, -1)
 new_values = np.dot(trans_matrix, vec).flatten()
 first_data_point_x[0, :4] = new_values[:4]


 



 vec = np.array([
    [0],[0],[0],[0]
  ])

 first_data_point_y = np.array([y[user_data[user_num][0]]])
 previous_class_idx = m[y[0]]
 vec[0][previous_class_idx] = 1
     
 # Initialize the active learner
 learner = ActiveLearner(
     estimator=RandomForestClassifier(criterion='log_loss'),
 )
 learner.fit(first_data_point_x, first_data_point_y)
 
 x_train, x_test = train_test_split (x[user_data[user_num][0]:user_data[user_num][1]],test_size=0.20, random_state=42, shuffle = False)
 y_train,y_test = train_test_split (y[user_data[user_num][0]:user_data[user_num][1]],test_size=0.20, random_state=42, shuffle =False)
 # Road parameters
 epsilon =0.1

 learning_rate = 0.01
 threshold = 0.9
 positive_reward = 1.0
 negative_reward = -2.0
 probe_count = 1

 for idx, (data_point, true_label) in enumerate(zip(x_train, y_train)):
     # Make predictions
     new_values = np.dot(trans_matrix, vec).flatten()
     data_point[:4] = new_values[:4]
     prediction_probabilities = learner.predict_proba(data_point.reshape(1, -1))

     
     predicted_class = learner.predict(data_point.reshape(1, -1))[0]
     predicted_class_idx = m[predicted_class]
    
     max_probability = np.max(prediction_probabilities)
    
     rand = random.uniform(0, 1)
     if rand <= epsilon:
         predicted_class_idx = m[true_label]
         learner.teach(data_point.reshape(1, -1), np.array([true_label]))  # Wrap true_label in an array
         probe_count += 1
 

     else:
         # Check uncertainty and query for label if uncertain
         if max_probability <= threshold:
             predicted_class_idx = m[true_label]
             
             # Teach the model with the new labeled data
             learner.teach(data_point.reshape(1, -1), np.array([true_label]))  # Wrap true_label in an array
             probe_count += 1
             
                     # Calculate reward and update policy
             if predicted_class == true_label:
                 reward = negative_reward
             else:
                 reward = positive_reward
             
             threshold = min(threshold * (1 + learning_rate * (1 - pow(2, reward / negative_reward))), 1.0)
             
     vec = np.zeros((4, 1))
     vec[predicted_class_idx][0] = 1
     trans_freq[previous_class_idx][predicted_class_idx]+=1
     
     freq[previous_class_idx][0] +=1 
     for i in range(4):
       trans_matrix[previous_class_idx][i] = float (trans_freq[previous_class_idx][i]) /float (freq[previous_class_idx])
     previous_class_idx = predicted_class_idx     
     
 y_pred = learner.predict(x_test)
 print (trans_matrix)
 print (freq)
 print (trans_freq)
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