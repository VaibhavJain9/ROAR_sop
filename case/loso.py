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


x = df[['Score', 'arousal_acc_video', 'valence_acc_video', 'GSR_Diff']].values
y = df[['Probe']].values

f1_score = []
TPR_list = []
FPR_list = []

for i in range(30):
    print('user no:', i)
    l = 243*i
    r = 243*i + 243
    if i == 0:
        x_train = x[243:243*30]
        y_train = y[243: 243*30]
    elif i == 27:
        x_train = x[0:243*29]
        y_train = y[0: 243*29] 
    else:
     
        x_train = np.concatenate([x[0:243*i], x[243*(i+1):243*30]])
        y_train = np.concatenate([y[0:243*i], y[243*(i+1):243*30]])

    x_test = x[243*i: 243*(i+1)]
    y_test = y[243*i: 243*(i+1)]
    # Assuming x_test and y_test are numpy arrays or lists
    # Determine the index where the last 20% starts
    split_index = int(len(x_test) * 0.8)
    
    # Extract the last 20% of x_test and y_test
    x_test = x_test[split_index:]
    y_test = y_test[split_index:]

    logistic_classifier = LogisticRegression( class_weight='balanced')

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
