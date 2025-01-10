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


def plot_loss(ax, epochs, validation_loss, training_loss):
    # fig, ax = plt.subplots(1, 1, figsize=(10, 6),squeeze=False)
    
    ax.plot(epochs, training_loss + validation_loss, label = r"Total", zorder= 1, color = "k")
    ax.plot(epochs, training_loss, label = r"Training",zorder = 1, color = "tab:red")
    ax.plot(epochs, validation_loss, label = r"Validation",zorder = 1, color = "tab:blue")
    
    ax.set_ylabel(r"Loss")
    # ax.set_xlabel("Epoch")
    ax.legend(frameon=False, ncol=1, bbox_to_anchor=(1, 1), loc='upper left')
    
    return ax

def plot_confusion(ax,  epochs, confusion_matrix, true_class_label = 0):
    
    
    
    
    true_positives = np.diagonal(confusion_matrix, axis1=1, axis2=2)
    totals = confusion_matrix.sum(axis=2)
    recalls = np.divide(true_positives, totals, out=np.zeros_like(true_positives, dtype=float), where=totals!=0)
    
    false_positives = confusion_matrix.sum(axis=1) - true_positives
    precisions = np.divide(true_positives, true_positives + false_positives, out=np.zeros_like(true_positives, dtype=float), where=(true_positives + false_positives) != 0)
    
    accuracies = np.divide(true_positives.sum(axis=1), totals.sum(axis=1), out=np.zeros_like(true_positives.sum(axis=1), dtype=float), where=totals.sum(axis=1) != 0)
    
    
    
    ax.plot(epochs, recalls[:, true_class_label], label=r"Recall", zorder=1, color="tab:red")
    ax.plot(epochs, precisions[:, true_class_label], label=r"Precision", zorder=1, color="tab:blue")
    ax.plot(epochs, accuracies, label=r"Accuracy", zorder=1, color="tab:green")
    
    ax.set_ylabel(r"Metrics")
    # ax.set_xlabel("Epoch")
    ax.legend(frameon=False, ncol=1, bbox_to_anchor=(1, 1), loc='upper left')
    
    
    return ax