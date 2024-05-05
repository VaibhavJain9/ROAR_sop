import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # Import Logistic Regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
import statistics as stat
import numpy as np
from sklearn.metrics import confusion_matrix


# Read data
df = pd.read_csv('Total_features.csv')
df.reset_index(inplace=True)
df = df.sort_values(by=['P_id', 'index'])
df.drop(columns=['index'], inplace=True)


# Define features (x) and target (y)
x = df[['Score', 'EDA_diff', 'HR_diff', 'TEMP_diff', 'BVP_diff']].values
y = df[['Probe']].values


# Load user data
user_data = {
    'start': [0, 313, 508, 819, 1073, 1369, 1668, 1912, 2154, 2374, 2594, 2802, 3006, 3206, 3494, 3834, 4169, 4361, 4537, 4730, 4917, 5100, 5274, 5530, 5788, 6005, 6204, 6419],
    'end': [313, 508, 819, 1073, 1369, 1668, 1912, 2154, 2374, 2594, 2802, 3006, 3206, 3494, 3834, 4169, 4361, 4537, 4730, 4917, 5100, 5274, 5530, 5788, 6005, 6204, 6419, 6644]
}


f1_score = []
TPR_list = []
FPR_list = []

for i in range(28):
    print('user no:', i)
    if i == 0:
        x_train = x[user_data['start'][i+1]:user_data['end'][27]]
        y_train = y[user_data['start'][i+1]:user_data['end'][27]]
    elif i == 27:
        x_train = x[user_data['start'][0]:user_data['end'][i-1]]
        y_train = y[user_data['start'][0]:user_data['end'][i-1]] 
    else:
        start_prev = user_data['end'][i]
        start_next = user_data['start'][i+1]
     
        x_train = np.concatenate([x[user_data['start'][0]:start_prev], x[start_next:user_data['end'][27]]])
        y_train = np.concatenate([y[user_data['start'][0]:start_prev], y[start_next:user_data['end'][27]]])

    x_test = x[user_data['start'][i]:user_data['end'][i]]
    y_test = y[user_data['start'][i]:user_data['end'][i]]
    
    # Assuming x_test and y_test are numpy arrays or lists
    # Determine the index where the last 20% starts
    split_index = int(len(x_test) * 0.8)
    
    # Extract the last 20% of x_test and y_test
    x_test = x_test[split_index:]
    y_test = y_test[split_index:]

    logistic_classifier = LogisticRegression()

    # Train the classifier on the training data
    logistic_classifier.fit(x_train, y_train)

    # Predict the labels of the test set
    probabilities = logistic_classifier.predict_proba(x_test)

# G et the predicted class labels using a threshold of 0.5
    predicted_labels = (probabilities[:, 1] > 0.3).astype(int)
    #y_pred = logistic_classifier.predict(x_test)
    y_pred = predicted_labels


    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
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
