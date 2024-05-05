import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from modAL.models import ActiveLearner
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt

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

f1_score= []
probe_ratio = []

# Load the CSV file into a DataFrame with semicolon (;) as the delimiter
df1 = pd.read_csv('user_data.csv', delimiter=';')

# Extract "start" and "End" columns , start and end are index for each user entries
user_data = df1[['start', 'End']].values 

total_count =0

for i in range (32):
 user_num = i
 print ("user number: " , user_num)
 happy =0
 sad =0
 relaxed =0 
 stressed =0 
 
 # Initialize the active learner
 # Initialize the active learner and fit it with the first data point
 x_t = x[user_data[user_num][0]:user_data[user_num][1]]
 y_t = y[user_data[user_num][0]:user_data[user_num][1]]
 learner = ActiveLearner(
     estimator=RandomForestClassifier(),
 )
 X_train, X_temp, y_train, y_temp = train_test_split(x_t, y_t, test_size=0.6, random_state=42, shuffle=False)


# Step 2: Split temp into active learning (40%) and test (20%)
 X_active, X_test, y_active, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42, shuffle=False)
 learner.fit(X_train, y_train)
 threshold = 0.8

 probe_count =0 
 probe_count+=X_train.shape[0]
 unique_values, counts = np.unique(y_train, return_counts=True)

# Update counters based on the frequency
 for value, count in zip(unique_values, counts):
    if value == -2:
        sad += count
    elif value == 0:
        relaxed += count
    elif value == 1:
        stressed += count
    elif value == 2:
        happy += count
 print(happy, sad, relaxed, stressed)
 for idx, (data_point, true_label) in enumerate(zip(X_active, y_active)):
     # Make predictions
     prediction_probabilities = learner.predict_proba(data_point.reshape(1, -1))
     
     predicted_class = learner.predict(data_point.reshape(1, -1))[0]

     max_probability = np.max(prediction_probabilities)
   #  print (threshold)

    # print (max_probability )
    # print (prediction_probabilities, predicted_class, true_label, threshold)
    
         # Check uncertainty and query for label if uncertain
     if max_probability <= threshold:
         
         # Teach the model with the new labeled data
         learner.teach(data_point.reshape(1, -1), np.array([true_label]))  # Wrap true_label in an array
         if(true_label==-2) :
          sad +=1
         if(true_label==0) :
             relaxed +=1
         if(true_label==1):
             stressed +=1
         if(true_label==2):
             happy +=1

         probe_count += 1
         
         
 y_pred = learner.predict(X_test)
 

 precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')

 f1_score.append(f1)
 probe_ratio.append(probe_count/ ( user_data[user_num][1] -user_data[user_num][0]))
 
 # Print the results

 total_count += probe_count
 #print( " probe count :", probe_count, "number of samples :" , user_data[user_num][1] -user_data[user_num][0]  )
print(total_count)


user_labels = [f" {i + 1}" for i in range(len(f1_score))]
red_probes =[ (.8-score)/.8 for score in probe_ratio]
print (np.mean(f1_score))
user_labels = [f"{i + 1}" for i in range(len(f1_score))]

for i, value in enumerate(f1_score, start=1):
    print(f"User{i}: {value:.2f}")

plt.bar(user_labels, f1_score, color='blue')
plt.xlabel('Users')
plt.ylabel('F1 Score')
plt.title('F1 Score for Each User')
# plt.show()

# plt.bar(user_labels ,red_probes, color='orange')
# plt.xlabel('Users')
# plt.ylabel('Reduction in Probe Rate')
# plt.title('Reduction in Probe Rate for Each User')
# plt.show()

# print (sad, relaxed, happy , stressed)
# sad = (498-sad)/498
# relaxed = (4178-relaxed)/4178
# stressed = (1615 -stressed)/1615
# happy = (1235 -happy)/1235
# categories = ['Relaxed', 'Stressed', 'Happy', 'Sad']
# values = [relaxed, stressed, happy, sad]

# plt.bar(categories, values)
# plt.xlabel('Emotions')
# plt.ylabel('Probe reduction rate')
# plt.title('Bar Plot of Discrete Variables')
# plt.show()
