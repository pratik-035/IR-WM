
import numpy as np 
from sklearn.datasets import fetch_20newsgroups 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import MultinomialNB 

from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score, classification_report 

# Load the 20 Newsgroups dataset with selected categories 
 
categories = ['alt.atheism', 'rec.motorcycles', 'comp.graphics', 
'sci.med']

newsgroups = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

# Convert text data to numerical features using TF-IDF 
vectorizer = TfidfVectorizer(stop_words='english') 
X = vectorizer.fit_transform(newsgroups.data) 
y = newsgroups.target 

# Split dataset into training (80%) and testing (20%) sets 
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
random_state=42) 

# Train and evaluate Naïve Bayes classifier 
nb_classifier = MultinomialNB() 
nb_classifier.fit(X_train, y_train) 
y_pred_nb = nb_classifier.predict(X_test)

# Train and evaluate SVM classifier 
svm_classifier = SVC(kernel='linear') 
svm_classifier.fit(X_train, y_train) 
y_pred_svm = svm_classifier.predict(X_test) 

# Print evaluation results 
print("\n=== Naïve Bayes Classifier Performance ===") 
 
print(f"Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred_nb, 
target_names=categories)) 

print("\n=== Support Vector Machine (SVM) Classifier Performance ===") 
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}") 
print("Classification Report:\n", classification_report(y_test, 
y_pred_svm, target_names=categories)) 