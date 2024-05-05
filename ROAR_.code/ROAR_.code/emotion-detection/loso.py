import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
import numpy as np 

df = pd.read_csv('data.csv')
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
# y_pred_total stores all the predictions combined

# Load the CSV file into a DataFrame with semicolon (;) as the delimiter
df1 = pd.read_csv('user_data.csv', delimiter=';')

# Extract "start" and "End" columns , start and end are index for each user entries
user_data = df1[['start', 'End']].values 

y_pred_total = []
y_test_total =[]
f1_score = []


for i in range (32):
    print('user no:' , i)
    if i == 0:
     x_train = x[user_data[i+1][0]:user_data[31][1]]
     y_train = y[user_data[i+1][0]:user_data[31][1]]
    elif i == 31:
     x_train = x[user_data[0][0]:user_data[30][1]]
     y_train = y[user_data[0][0]:user_data[30][1]] 
    else:
     df1 = pd.DataFrame(x[user_data[0][0]:user_data[i-1][1]])
     df2 = pd.DataFrame(x[user_data[i+1][0]:user_data[31][1]])
     frames = [df1,df2]
     x_train = np.concatenate(frames) 
     df1 = pd.DataFrame(y[user_data[0][0]:user_data[i-1][1]])
     df2 = pd.DataFrame(y[user_data[i+1][0]:user_data[31][1]])
     frames = [df1,df2]
     
     y_train = np.concatenate (frames)
     
  

    x_test = x[user_data[i][0]:user_data[i][1]]
    y_test = y[user_data[i][0]:user_data[i][1]]
    # Assuming x_test and y_test are numpy arrays or lists

# Determine the index where the last 20% starts
    split_index = int(len(x_test) * 0.8)
    
    # Extract the last 20% of x_test and y_test
    x_test = x_test[split_index:]
    y_test = y_test[split_index:]



    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

 # Train the classifier on the training data
    rf_classifier.fit(x_train, y_train)
    
    # Make predictions on the test data
    y_pred = rf_classifier.predict(x_test)
    
    
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    f1_score.append(f1)
    




# this is weighted average of all the predictions 
 


print (np.mean(f1_score))
user_labels = [f"{i + 1}" for i in range(len(f1_score))]
for i, value in enumerate(f1_score, start=1):
    print(f"User {i}: {value:.2f}")
plt.bar(user_labels, f1_score, color='blue')
plt.xlabel('Users')
plt.ylabel('F1 Score')
plt.title('F1 Score for Each User')
# plt.show()


