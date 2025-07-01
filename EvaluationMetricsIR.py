
from collections import defaultdict
import numpy as np


def calculate_precision(tp, fp):
    """ 

    Calculates Precision: Precision = TP / (TP + FP) 

    :param tp: True Positives (correctly retrieved relevant documents) 

    :param fp: False Positives (incorrectly retrieved non-relevant 
    documents) 

    :return: Precision value 
    """

    return 0.0 if tp + fp == 0 else tp / (tp + fp)


def calculate_recall(tp, fn):

    """ 
    Calculates Recall: Recall = TP / (TP + FN) 
 
    :param tp: True Positives (correctly retrieved relevant documents) 
 
    :param fn: False Negatives (relevant documents not retrieved) 
 
    :return: Recall value 
    """
    return 0.0 if tp + fn == 0 else tp / (tp + fn)


def calculate_f1_score(precision, recall): 
    """ 
    Calculates F1-Score: F1 = 2 * (Precision * Recall) / (Precision + 
    Recall) 
    :param precision: Calculated Precision value 
 
    :param recall: Calculated Recall value 
    
    :return: F1-Score value 
    """ 
    
    return 0.0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall) 

def evaluate_retrieval(relevant_docs, retrieved_docs): 
    """ 
    Evaluates the retrieval performance by calculating TP, FP, and FN. 
 
    :param relevant_docs: A set of relevant document IDs (ground truth) 
 
    :param retrieved_docs: A set of document IDs retrieved by the IR 
    system 
 
    :return: Computed Precision, Recall, and F1-score 
    """ 
    
    tp = len(relevant_docs & retrieved_docs) # True Positives (Correctly retrieved relevant docs) 
    fp = len(retrieved_docs - relevant_docs) # False Positives (Retrieved but not relevant) 
 
    fn = len(relevant_docs - retrieved_docs) # False Negatives (Relevant but not retrieved) 
 
 
 
    # Calculate metrics 
 
    precision = calculate_precision(tp, fp) 
    recall = calculate_recall(tp, fn) 
    f1_score = calculate_f1_score(precision, recall) 
    
    return tp, fp, fn, precision, recall, f1_score


# Example Data: Ground Truth & Retrieved Documents 

relevant_documents = {1, 2, 3, 5, 7}  # Actual relevant document IDs 
retrieved_documents = {1, 2, 4, 5, 6}    # Documents retrieved by the IR system

# Compute evaluation metrics 
tp, fp, fn, precision, recall, f1_score = evaluate_retrieval(relevant_documents, retrieved_documents)


# Display Results 
print("\n=== Evaluation Metrics for Information Retrieval ===") 
print(f"True Positives (TP): {tp}") 
print(f"False Positives (FP): {fp}") 
print(f"False Negatives (FN): {fn}") 
print(f"Precision: {precision:.4f}") 
print(f"Recall: {recall:.4f}") 
print(f"F1-Score: {f1_score:.4f}") 



