import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn  as sk

def plot_confusion_matrix_correlation(confusion_matrix):
    """
    Plots a correlation heatmap for the accuracy of an n x n confusion matrix.
    
    Parameters:
    confusion_matrix (np.ndarray): The confusion matrix (n x n).
    """
    # Calculate the accuracy for each class
    sk.metrics.ConfusionMatrixDisplay(confusion_matrix,normalize=True)
    accuracy = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)
    
    # Create a correlation matrix
    correlation_matrix = np.outer(accuracy, accuracy)
    
    # Plot the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=range(confusion_matrix.shape[0]), yticklabels=range(confusion_matrix.shape[0]))
    plt.title("Correlation Plot for Accuracy of Confusion Matrix")
    plt.xlabel("Class")
    plt.ylabel("Class")
    plt.show()

# Example usage
# confusion_matrix = np.array([[50, 2, 1], [10, 40, 5], [3, 4, 60]])
# plot_confusion_matrix_correlation(confusion_matrix)