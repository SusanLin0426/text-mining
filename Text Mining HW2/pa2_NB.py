import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve
import matplotlib.pyplot as plt


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

# Split the training dataset into training (90%) and evaluation (10%) sets with stratified sampling
X_train, X_eval, y_train, y_eval = train_test_split(train_df['text'], train_df['label'], test_size=0.1, random_state=3, stratify=train_df['label'])

training_data_index_flat_sorted = sorted(training_data_index_flat)
df_sorted = df.sort_values(by='label')

# Use the remaining documents as the testing dataset, excluding those in training_data_index
# test_df = df_sorted[~df_sorted.index.isin(training_data_index_flat_sorted)]
test_df = df_sorted[~df_sorted['label'].isin(training_data_index_flat_sorted)]
# X_test, y_test = test_df['text'], test_df['label']
X_test = test_df['text']

sorted_test_df = test_df.sort_values(by='label')

# Convert documents to multi-hot vectors for training set
vectorizer = CountVectorizer(binary=True)
X_train_vectorized = vectorizer.fit_transform(X_train)

# Convert documents to multi-hot vectors for evaluation and test sets
X_eval_vectorized = vectorizer.transform(X_eval)
X_test_vectorized = vectorizer.transform(X_test)

# Train the Bernoulli Naive Bayes model
bnb_model = BernoulliNB()
bnb_model.fit(X_train_vectorized, y_train)

# Predict on the evaluation set
predicted_labels_eval = bnb_model.predict(X_eval_vectorized)

# Display the classification report for the evaluation set
print("Classification Report for Evaluation Set:")
print(classification_report(y_eval, predicted_labels_eval))

# Plot the Precision-Recall curve for the evaluation set
# Plot the Precision-Recall curve for each class
precision_eval = dict()
recall_eval = dict()

for i in range(1, 14):
    binary_y_eval = label_binarize(y_eval, classes=list(range(1, 14)))[:, i - 1]
    precision_eval[i], recall_eval[i], _ = precision_recall_curve(binary_y_eval, bnb_model.predict_proba(X_eval_vectorized)[:, i - 1])
    plt.plot(recall_eval[i], precision_eval[i], label=f'Class {i}')

# Show the plot for evaluation
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for each class (Evaluation)')
plt.legend()
plt.show()


# Predict on the testing set
predicted_labels_test = bnb_model.predict(X_test_vectorized)

# Create a DataFrame for the predicted results
result_df = pd.DataFrame({'Id': test_df['label'], 'Value': predicted_labels_test})
# Sort the DataFrame by the 'Id' column
result_df = result_df.sort_values(by='Id')
# Save the DataFrame to a CSV file
result_df.to_csv('predicted_results_NB.csv', index=False)