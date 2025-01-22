import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn  as sk
import modelframe as mf
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_recall_matrix(frame : mf.GlitchModel):
    matrix : np.ndarray = frame.con_matrix_per_epoch[-1]
    fig, ax = plt.subplots(1, 1, squeeze=False)
    heatmap = sns.heatmap(matrix/np.sum(matrix, axis = 0), 
                ax=ax[0,0],annot=True, fmt=".2f", cmap = 'coolwarm', xticklabels=range(matrix.shape[0]), yticklabels=range(matrix.shape[1]),cbar=False, vmin = 0, vmax = 1)
    for row in range(ax.shape[0]):
        for col in range(ax.shape[1]):
            ax[row,col].set_xlabel("Predicted")
            ax[row,col].set_ylabel("True")
            ax[row,col].set_aspect('equal')
            
    divider = make_axes_locatable(ax[0, 0])
    cax = divider.append_axes("right", size = "5%")
    colorbar = fig.colorbar(heatmap.get_children()[0], cax=cax)
    colorbar.set_label("Recall", rotation=270, labelpad=15)
    
    return fig, ax

def plot_accuracy_matrix(frame: mf.GlitchModel):
    matrix : np.ndarray = frame.con_matrix_per_epoch[-1]
    fig, ax = plt.subplots(1, 1, squeeze=False)
    totalacc = np.trace(matrix)/np.sum(matrix)
    heatmap = sns.heatmap(matrix/np.sum(matrix), 
                ax=ax[0,0],annot=True, fmt=".2f", cmap = 'coolwarm', xticklabels=range(matrix.shape[0]), yticklabels=range(matrix.shape[1]),cbar=False, vmin = 0, vmax = 1)
    for row in range(ax.shape[0]):
        for col in range(ax.shape[1]):
            ax[row,col].set_xlabel("Predicted")
            ax[row,col].set_ylabel("True")
            ax[row,col].set_aspect('equal')
            
    divider = make_axes_locatable(ax[0, 0])
    cax = divider.append_axes("right", size = "5%")
    colorbar = fig.colorbar(heatmap.get_children()[0], cax=cax)
    colorbar.set_label("Class Accuracy", rotation=270, labelpad=15)
    ax[0,0].set_title(f"Total accuracy: {totalacc:.2f}")
    
    return fig, ax

def plot_precision_matrix(frame: mf.GlitchModel):
    matrix : np.ndarray = frame.con_matrix_per_epoch[-1]
    fig, ax = plt.subplots(1, 1, squeeze=False)
    heatmap = sns.heatmap((matrix.T/np.sum(matrix.T, axis = 0)).T, 
                ax=ax[0,0],annot=True, fmt=".2f", cmap = 'coolwarm', xticklabels=range(matrix.shape[0]), yticklabels=range(matrix.shape[1]),cbar=False, vmin = 0, vmax = 1)
    for row in range(ax.shape[0]):
        for col in range(ax.shape[1]):
            ax[row,col].set_xlabel("Predicted")
            ax[row,col].set_ylabel("True")
            ax[row,col].set_aspect('equal')
            
    divider = make_axes_locatable(ax[0, 0])
    cax = divider.append_axes("right", size = "5%")
    colorbar = fig.colorbar(heatmap.get_children()[0], cax=cax)
    colorbar.set_label("Precision", rotation=270, labelpad=15)
    
    return fig, ax


def plot_confusion_matrix_correlation(frame : mf.GlitchModel):
    
    matrix : np.ndarray = frame.con_matrix_per_epoch[-1]
    fig, ax = plt.subplots(1, 3, squeeze=False)
    totalacc = np.trace(matrix)/np.sum(matrix)
    
    
    
    heatmap1 = sns.heatmap(matrix/np.sum(matrix, axis = 0), 
                ax=ax[0,0],annot=True, fmt=".2f", cmap = 'coolwarm', xticklabels=range(matrix.shape[0]), yticklabels=range(matrix.shape[1]),cbar=False, vmin = 0, vmax = 1)
    heatmap2 = sns.heatmap((matrix.T/np.sum(matrix.T, axis = 0)).T, 
                ax=ax[0,1],annot=True, fmt=".2f", cmap = 'coolwarm', xticklabels=range(matrix.shape[0]), yticklabels=range(matrix.shape[1]),cbar=False, vmin = 0, vmax = 1)
    heatmap3 = sns.heatmap(matrix/np.sum(matrix), 
                ax=ax[0,2],annot=True, fmt=".2f", cmap = 'coolwarm', xticklabels=range(matrix.shape[0]), yticklabels=range(matrix.shape[1]),cbar=False, vmin = 0, vmax = 1)
    ax[0,0].set_title("Recall")
    ax[0,1].set_title("Precision")
    ax[0,2].set_title(f"Class Accuracy, Total accuracy: {totalacc:.2f}")
    
    
    
    
    for row in range(ax.shape[0]):
        for col in range(ax.shape[1]):
            ax[row,col].set_xlabel("Predicted")
            ax[row,col].set_ylabel("True")
            ax[row,col].set_aspect('equal')
            
    divider = make_axes_locatable(ax[0, 2])
    cax = divider.append_axes("right", size = "5%")
    colorbar = fig.colorbar(heatmap3.get_children()[0], cax=cax)
    colorbar.set_label("Normalized Value", rotation=270, labelpad=15)
    
    return fig, ax

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


from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

def plot_performance(frame):
    epochs = np.arange(1, frame.number_of_epochs+1, 1)


    fig, ax = plt.subplots(2, 1, squeeze=  False, figsize = (10, 4),sharex=True)
    ax[0,0].plot(epochs, frame.training_loss + frame.validiation_loss, label = r"Total Loss", zorder= 1, color = "k")
    ax[0,0].plot(epochs, frame.training_loss, label = r"Training Loss",zorder = 1, color = "tab:red")
    ax[0,0].plot(epochs, frame.validiation_loss, label = r"Validation Loss",zorder = 1, color = "tab:blue")
    ax[1,0].plot(epochs, frame.accuracy, label = r"Accuracy",zorder = 1, color = "k")
    ax[1,0].plot(epochs, frame.precision, label = r"Precision",zorder = 1, color = "tab:red")
    ax[1,0].plot(epochs, frame.recall, label = r"Recall",zorder = 1, color = "tab:blue")

    ax[1,0].axhline(frame.test_accuracy, label = r"Accuracy(test)", zorder = 1, color = "k", linestyle = "dashed")
    ax[1,0].axhline(frame.test_precision, label = r"Precision(test)", zorder = 1, color = "tab:red", linestyle = "dashed")
    ax[1,0].axhline(frame.test_recall, label = r"Recall(test)", zorder = 1, color = "tab:blue", linestyle = "dashed")

    # ax[0,0].axhline((total_loss).mean(),zorder= 0,color = "k", linestyle = "dotted")
    # ax[0,0].axhline((total_accuracy).mean(),zorder= 0,color = "k", linestyle = "dotted")
    ax[0,0].set_ylabel(r"Value")
    ax[1,0].set_ylabel(r"Value")
    for i in range(2):
        ax[i,0].tick_params(axis="y",direction="in",which="both")
        ax[i,0].tick_params(axis="x",direction="in",which="both")
        # ax[i,0].set_yticks(np.arange(0, np.max(total_loss), int(np.max(total_loss)) /10))
        ax[i,0].set_xticks(np.arange(0, frame.number_of_epochs + 1, frame.number_of_epochs // 10))
        ax[i,0].xaxis.set_minor_locator(MultipleLocator(frame.number_of_epochs//50))
        ax[i,0].yaxis.set_minor_locator(AutoMinorLocator())
    ax[1,0].set_xlabel(r"Epoch")
    # ax[0,0].set_ylabel("Value")
    ax[0,0].set_xlim(0, frame.number_of_epochs)
    ax[0,0].set_ylim(0, np.max((frame.training_loss + frame.validiation_loss)[~np.isnan(frame.training_loss + frame.validiation_loss)]))
    ax[1,0].set_ylim(0, 1.2)
    ax[0,0].legend(frameon=False, ncol=1, bbox_to_anchor=(1, 1), loc='upper left')
    ax[1,0].legend(frameon=False, ncol=1, bbox_to_anchor=(1, 1), loc='upper left')
    fig.subplots_adjust(hspace=0.1)
    return fig, ax




# ax[0,0].grid(True,which="minor")