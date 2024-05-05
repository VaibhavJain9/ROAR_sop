import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
from modAL.models import ActiveLearner
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
import statistics
import random

df = pd.read_csv('Total_features.csv')

df.reset_index(inplace=True)

# Sorting by 'P_id' and then by index column
df = df.sort_values(by=['P_id', 'index'])

# Dropping the added index column if necessary
df.drop(columns=['index'], inplace=True)

list1 = []

# print (df)

x = df[['Score', 'EDA_diff', 'HR_diff', 'TEMP_diff', 'BVP_diff']].values
y = df[['Probe']].values

# print (x)
# print (y)

scores = []


# Load the CSV file into a DataFrame with semicolon (;) as the delimiter
df = pd.read_csv("user_data.csv", delimiter=",")

# Select only the 'start' and 'end' columns
user_data = df[['start', 'end']]

#
for j in [0.1, 0.2, 0.3, 0.4, 0.5 ,0.6, 0.7 , 0.8, 0.9]:


 f1_scores =[]
 TPR_list = []
 FPR_list = []
 class_1_red  = []
 class_0_red = []
 red = []
 total_count = 0 
 for i in range(len(user_data)):
  user_num = i
 
  print ("user number: " , user_num)
  
  l = user_data.loc[i, 'start']
  r = user_data.loc[i, 'end']
  
  # Initialize the active learner
  # Initialize the active learner and fit it with the first data point
  x_t = x[l:r]
  y_t = y[l:r]
 
  # Initialize the active learner
  learner = ActiveLearner(
       estimator=LogisticRegression( ), 
  )
  X_start, X_temp, y_start, y_temp = train_test_split(x_t, y_t, test_size=0.80, random_state=42, shuffle=False)
  learner.fit(X_start, y_start)
  
  x_train, x_test = train_test_split (X_temp,test_size=0.25, random_state=42, shuffle = False)
  y_train,y_test = train_test_split (y_temp,test_size=0.25, random_state=42, shuffle =False)
  # Road parameters
  epsilon = 0.2
  learning_rate = 0.05
  zero_count =0 
  one_count =0 
  threshold = j
  positive_reward = 1.0
  negative_reward = -2.0

  probe_count = X_start.shape[0]
  one_count = sum (y_start)
  zero_count = X_start.shape[0] - one_count
 
  for idx, (data_point, true_label) in enumerate(zip(x_train, y_train)):
      # Make predictions
      
      prediction_probabilities = learner.predict_proba(data_point.reshape(1, -1))
 
      
      predicted_class = learner.predict(data_point.reshape(1, -1))[0]
 
     
      max_probability = np.max(prediction_probabilities)
     
      rand = random.uniform(0, 1)
      if rand <= epsilon:
         
          learner.teach(data_point.reshape(1, -1), np.array([true_label]))  # Wrap true_label in an array
          probe_count += 1

          if true_label ==1 :
             one_count += 1
          else :
             zero_count += 1
 
      else:
          # Check uncertainty and query for label if uncertain
          if max_probability <= threshold:
                # Teach the model with the new labeled data
              learner.teach(data_point.reshape(1, -1), np.array([true_label]))  # Wrap true_label in an array
              probe_count += 1
              
              if true_label == 1 :
                one_count += 1
              else :
                zero_count += 1 
                
                      # Calculate reward and update policy
              if predicted_class == true_label:
                  reward = negative_reward
              else:
                  reward = positive_reward
              
              threshold = min(threshold * (1 + learning_rate * (1 - pow(2, reward / negative_reward))), 1.0)
              
   
      
  #y_pred = learner.predict(x_test)
  probabilities = learner.predict_proba(x_test)
 
 # Get the predicted class labels using a threshold of 0.5
  predicted_labels = (probabilities[:, 1] > 0.35).astype(int)
    #y_pred = logistic_classifier.predict(x_test)
  y_pred = predicted_labels            
 
#   print ('threshold', threshold)
 
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
  
  TPR_list.append (TPR)
  FPR_list.append (FPR)
  # Print the results
  f1_scores.append(f1)
  one_sum = sum (y_t)
  red.append ((.8 - probe_count/x_t.shape[0])/.8)
  class_0_red.append ((.8 - zero_count/(x_t.shape[0]-one_sum))/.8)
  class_1_red.append ((.8 - one_count/one_sum)/.8)

 
  total_count += probe_count
  
 
 print(f1_scores)
 print (threshold)
 print (np.mean(f1_scores))
 print ("TPR: " , np.mean (TPR_list))
 print ("FPR: " , np.mean (FPR_list))
 print (total_count)
 
 mean_TPR = np.mean(TPR_list)
 mean_FPR = np.mean(FPR_list)
 mean_f1 = np.mean(f1_scores)

 list_temp = [j, mean_f1,statistics.stdev (f1_scores),  mean_TPR, statistics.stdev(TPR_list),  mean_FPR, statistics.stdev(FPR_list) 
              , np.mean (red),
       statistics.stdev(red) , np.mean (class_0_red)
       , np.mean (class_1_red )]
 list1.append (list_temp)
print (list1) 
#print (sum(zero_count))

# Append to scores list