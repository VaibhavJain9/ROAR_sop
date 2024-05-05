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
import copy
import random

df = pd.read_csv('Total_features.csv')

list1 = []

x = df[['Score', 'arousal_acc_video', 'valence_acc_video', 'GSR_Diff']].values
y = df[['Probe']].values
z = df[['video']].values


probe_rate = []
# print (x)
# print (y)
f1_score =[]
TPR_list = []
FPR_list = []
class_1_red  = []
class_0_red = []
red = []
total_count = 0 




# for i in range (len(user_data)):
#  l =  user_data.loc[i, 'start']
#  r = user_data.loc[i, 'End']
#  x_t = x[l:r]
#  y_t = y[l:r]
 
#  X_train, X_temp, y_train, y_temp = train_test_split(x_t, y_t, test_size=0.6, random_state=42, shuffle=False)
#  if i==0:
#   base_learner.fit(X_train, y_train)
#  else :
#   base_learner.teach(X_train,y_train)
 

for i in range(30):
 user_num = i
 array = np.empty(8, dtype=int)
 array.fill(0)
 print ("user number: " , user_num)
 
 l = 243*i
 r= l+ 243
 
 # Initialize the active learner
 # Initialize the active learner and fit it with the first data point
 x_t = x[l:r]
 y_t = y[l:r]
 z_t = z[l:r]
 learner = ActiveLearner(
      estimator= LogisticRegression (class_weight= 'balanced', solver = 'newton-cholesky', C =10)   
 )

 X_train, X_temp, y_train, y_temp, z_train , z_temp = train_test_split(x_t, y_t, z_t, test_size=0.6, random_state=42, shuffle=False)
 
#  print (X_train)
#  print (X_temp)

# Step 2: Split temp into active learning (40%) and test (20%)
 X_active, X_test, y_active, y_test, z_active, z_test = train_test_split(X_temp, y_temp,z_temp, test_size=0.34, random_state=42, shuffle=False)
 learner.fit (X_train, y_train)
 result_tuple = np.column_stack((y_train, z_train))
 for yi,zi in result_tuple:
    if yi==1 :
       array[zi-1]+=1
 




 threshold = 0.5
 probe_count =0
 zero_count = 0
 one_count= 0
 probe_count+=X_train.shape[0]
 one_count = sum (y_train)
 zero_count = X_train.shape[0] - one_count


 for idx, (data_point, true_label, video_id) in enumerate(zip(X_active, y_active, z_active)):
    
   #  print (data_point)
    # print (true_label)
     prediction_probabilities = learner.predict_proba(data_point.reshape(1, -1))
     
     predicted_class = learner.predict(data_point.reshape(1, -1))[0]

     max_probability = np.max(prediction_probabilities)

     if max_probability <= threshold:
         
         # Teach the model with the new labeled data
         learner.teach(data_point.reshape(1, -1), np.array([true_label]))  # Wrap true_label in an array

         probe_count += 1
         if true_label == 0:
            zero_count +=1
         else :
            one_count +=1 
            array[video_id-1]+=1
            
         
         
#  y_pred = learner.predict(X_test)
 probabilities = learner.predict_proba(X_test)

# Get the predicted class labels using a threshold of 0.5
 predicted_labels = (probabilities[:, 1] > 0.30).astype(int)
   #y_pred = logistic_classifier.predict(x_test)
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
 TPR_list.append (TPR)
 FPR_list.append (FPR)
 probe_rate.append(np.mean (array))
 one_sum = sum (y_t)
 red.append ((.8 - probe_count/x_t.shape[0])/.8)
 class_0_red.append ((.8 - zero_count/(x_t.shape[0]-one_sum))/.8)
 class_1_red.append ((.8 - one_count/one_sum)/.8)


 total_count += probe_count
 

print (f1_score)
print (threshold)
print (np.mean(f1_score))
print(total_count)




print ("TPR: " , np.mean (TPR_list))
print ("FPR: " , np.mean (FPR_list))
print ("Mean _ Red" , np.mean (red), np.mean (class_0_red), np.mean (class_1_red))
print ("std", statistics.stdev (f1_score), statistics.stdev (TPR_list), statistics.stdev (FPR_list))
print ("red" , (0.8 - total_count/8608)/.8)

print (threshold, np.mean (f1_score),statistics.stdev (f1_score),  np.mean (TPR_list),statistics.stdev (TPR_list), np.mean (FPR_list), statistics.stdev (FPR_list), np.mean (red),
       statistics.stdev(red) , np.mean (class_0_red)
       , np.mean (class_1_red ))
print (np.mean(probe_rate))

