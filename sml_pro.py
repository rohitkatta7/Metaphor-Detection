
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
#import tensorflow as tf
#from sklearn.preprocessing import LabelEncoder

# nltk.download('punkt')
# nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('train.csv')

# Dictionary linking metaphorID to specific words
metaphor_words = {
    0: 'road',
    1: 'candle',
    2: 'light',
    3: 'spice',
    4: 'ride',
    5: 'train',
    6: 'boat'
}

# Function to clean and extract context around the metaphor candidate word
def extract_context(text, metaphor_id, window_size=5):
    target_word = metaphor_words[metaphor_id]
    words = word_tokenize(text)
    if target_word in words:
        target_index = words.index(target_word)
        start = max(0, target_index - window_size)
        end = min(len(words), target_index + window_size + 1)
        return ' '.join(words[start:end])
    else:
        return ''

# Apply the cleaning and context extraction function
df['context'] = df.apply(lambda row: extract_context(row['text'], row['metaphorID']), axis=1)

# Remove rows with empty context
df = df[df['context'] != '']

# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = tfidf_vectorizer.fit_transform(df['context']).toarray()
y = df['label_boolean'].values

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "Naive Bayes": MultinomialNB()
}

# Perform cross-validation and compare models
k = 5  # Number of folds
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=k, scoring='accuracy')
    print(f"{name} - Cross-Validation Scores for {k} folds: {cv_scores}")
    print(f"{name} - Average Accuracy: {cv_scores.mean()}")

# PyTorch Dataset
class MetaphorDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

train_dataset = MetaphorDataset(X_train, y_train)
test_dataset = MetaphorDataset(X_test, y_test)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)  # Batch size set to 1 for stochastic gradient descent
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Neural Network Model
class MetaphorNN(nn.Module):
    def __init__(self, input_size):
        super(MetaphorNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # 2 for binary classification

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Model, Loss, Optimizer
model = MetaphorNN(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training Loop
for epoch in range(10):  # number of epochs
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Evaluation
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

accuracy_nn = evaluate_model(model, test_loader)
print(f"Accuracy: {accuracy_nn}")
'''
# Load the dataset
df = pd.read_csv('train.csv')

# ... [Include your data preprocessing steps here] ...

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Limiting to top 5000 features
X = tfidf_vectorizer.fit_transform(df['text']).toarray()
y = df['label_boolean'].values

# Encoding the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# TensorFlow Neural Network
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(X_train, y_train, epochs=6, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy_tf = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy_tf}")
'''
import matplotlib.pyplot as plt

logistic_accuracy = 0.836  # Example value
random_forest_accuracy = 0.861  # Example value
svm_accuracy = 0.845  # Example value
naive_bayes_accuracy = 0.831  # Example value
accuracy_nn  # Example value
#accuracy_tf
# Storing the accuracies in a dictionary
accuracies = {
    "Logistic Regression": logistic_accuracy,
    "Random Forest": random_forest_accuracy,
    "SVM": svm_accuracy,
    "Naive Bayes": naive_bayes_accuracy,
    "Neural Network (PyTorch)": accuracy_nn
    #"Neural Network (TensorFlow)": accuracy_tf
}

# Creating a bar chart
plt.figure(figsize=(10, 6))
plt.bar(accuracies.keys(), accuracies.values(), color='skyblue')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Comparison of Model Accuracies')
plt.ylim([min(accuracies.values()) - 0.05, max(accuracies.values()) + 0.05])  # Adjust the y-axis limits based on the accuracy range
plt.xticks(rotation=45)
plt.show()