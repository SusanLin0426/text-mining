import os
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

import os

# Get current path
script_dir = os.path.dirname(__file__)

# Specify the path to data folders
data_folder_path = os.path.join(script_dir, 'PA1-data')
txt_file_path = os.path.join(script_dir,'training_new.txt')


# Read 1095 text files
documents = []
labels = []

for filename in os.listdir(data_folder_path):
    with open(os.path.join(data_folder_path, filename), 'r', encoding='utf-8') as file:
        documents.append(file.read())

        # Extract the class ID from the file name without considering the extension
        class_id = int(filename.split('_')[0].replace('.txt', ''))
        labels.append(class_id)

# Create a DataFrame for easier handling
df = pd.DataFrame({'text': documents, 'label': labels})

# Read class and training data information
training_data_dict = {}
training_data_index = []
with open(txt_file_path, 'r') as file:
    for line in file:
        line = line.strip()
        elements = line.split(' ')

        class_id = int(elements[0])
        training_data = [int(idx) for idx in elements[1:]]

        training_data_dict[class_id] = training_data
        training_data_index.append(training_data)
# Convert the list of lists to a flat list
training_data_index_flat = [item for sublist in training_data_index for item in sublist]

# Create the training dataset
training_data = []

for class_id, data_list in training_data_dict.items():
    for idx in data_list:
        training_data.append({'text': df.iloc[idx]['text'], 'label': class_id})

# Convert training data to a DataFrame
train_df = pd.DataFrame(training_data)

# Split the training dataset into training (90%) and evaluation (10%) sets
X_train, X_eval, y_train, y_eval = train_test_split(train_df['text'], train_df['label'], test_size=0.1, random_state=3,stratify=train_df['label'])

training_data_index_flat_sorted = sorted(training_data_index_flat)
df_sorted = df.sort_values(by='label')
# Use the remaining documents as the testing dataset
test_df = df_sorted[~df_sorted['label'].isin(training_data_index_flat_sorted)]  # Exclude the training set indices
X_test = test_df['text']
sorted_test_df = test_df.sort_values(by='label')

# Initialize the TF-IDF vectorizer: Lowercase & Filter out English stopwords
tfidf_vectorizer = TfidfVectorizer(lowercase=True, stop_words=nltk.corpus.stopwords.words('english'))

# Get the TF-IDF vectors for training, evaluation, and testing sets
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_eval_tfidf = tfidf_vectorizer.transform(X_eval)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# SVM with RBF Kernel
svm_rbf = OneVsRestClassifier(SVC(kernel='rbf', probability=True))  # Set probability to True for decision_function
svm_rbf.fit(X_train_tfidf, y_train)

# Predict on the evaluation set and testing set
predicted_labels_eval_rbf = svm_rbf.predict(X_eval_tfidf)
predicted_labels_test_rbf = svm_rbf.predict(X_test_tfidf)

# Display the classification report for the evaluation set
print("SVM with RBF Kernel - Classification Report for Evaluation Set:")
print(classification_report(y_eval, predicted_labels_eval_rbf))

# # Get decision scores on the evaluation set
# decision_scores_eval_rbf = svm_rbf.decision_function(X_eval_tfidf)
# Get predicted probabilities on the evaluation set
probabilities_eval_rbf = svm_rbf.predict_proba(X_eval_tfidf)


# Binarize the labels for each class
y_eval_bin_rbf = label_binarize(y_eval, classes=np.unique(y_train))

# Initialize the plot
plt.figure(figsize=(8, 6))

# Plot precision-recall curve for each class on the evaluation set
for i in range(len(np.unique(y_train))):
    precision, recall, _ = precision_recall_curve(y_eval_bin_rbf[:, i], probabilities_eval_rbf[:, i])
    plt.plot(recall, precision, label=f'Class {i+1}')

# Set plot labels and title
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for SVM with RBF Kernel (Evaluation)')

# Show legend
plt.legend()

# Show the plot
plt.show()


# Create a DataFrame for the predicted results
result_df = pd.DataFrame({'Id': test_df['label'], 'Value': predicted_labels_test_rbf})

# Sort the DataFrame by the 'Id' column
result_df = result_df.sort_values(by='Id')

# Save the DataFrame to a CSV file
result_df.to_csv('predicted_results_svmRbf.csv', index=False)
