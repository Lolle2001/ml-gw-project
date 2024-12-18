import torch
import numpy as np
import numba as nb
import sklearn.metrics as mt

class ConfusionMatrix:
    def __init__(self,number_of_classes : int):
            self.number_of_class = number_of_classes
            self.predicted_class = np.array([])
            self.true_class = np.array([])
            self.confusion_matrix = np.zeros((number_of_classes, number_of_classes))
            self.accuracy= 0
            self.precision = 0
            self.recall= 0
    
    def calculate(self):
        # for true, pred in zip(self.true_class, self.predicted_class):
        #     confusion_matrix[true,pred] += 1

        self.confusion_matrix = mt.confusion_matrix(self.true_class, self.predicted_class)
        self.accuracy = mt.accuracy_score(self.true_class, self.predicted_class)
        self.recall = mt.recall_score(self.true_class, self.predicted_class)
        self.precision = mt.precision_score(self.true_class, self.predicted_class)

    def add(self, true: np.ndarray, predicted:np.ndarray):
        self.predicted_class = np.concatenate((self.predicted_class, predicted))
        self.true_class = np.concatenate((self.true_class, true))


# @nb.njit()
# def calculate_confusion_matrix_opt( 
#         number_of_classes : int,
#         predicted_class : np.ndarray, 
#         true_class: np.ndarray) -> np.ndarray:
    
#     confusion_matrix = np.zeros((number_of_classes, number_of_classes))
#     # if type(predicted_class) and type(true_class) == torch.tensor:
#     #     for ix in range(number_of_classes):
#     #         for iy in range(number_of_classes):
#     #             mask_predict = predicted_class  == iy
#     #             confusion_matrix[ix,iy] = (predicted_class[mask_predict]==ix).sum().item()
#     # elif type(predicted_class) and type(true_class) == np.ndarray:
#     for true, pred in zip(true_class, predicted_class):
#        confusion_matrix[true,pred] += 1
#     # else:
#         # print("Allowed types for class vectors are: torch.tensor and numpy.ndarray")
#     return confusion_matrix



# def calculate_confusion_matrix( 
#         number_of_classes : int,
#         predicted_class : np.ndarray, 
#         true_class: np.ndarray) -> np.ndarray:
    
#     confusion_matrix = np.zeros((number_of_classes, number_of_classes))
#     # if type(predicted_class) and type(true_class) == torch.tensor:
#     #     for ix in range(number_of_classes):
#     #         for iy in range(number_of_classes):
#     #             mask_predict = predicted_class  == iy
#     #             confusion_matrix[ix,iy] = (predicted_class[mask_predict]==ix).sum().item()
#     # elif type(predicted_class) and type(true_class) == np.ndarray:
#     for true, pred in zip(true_class, predicted_class):
#        confusion_matrix[true,pred] += 1
#     # else:
#         # print("Allowed types for class vectors are: torch.tensor and numpy.ndarray")
#     return confusion_matrix