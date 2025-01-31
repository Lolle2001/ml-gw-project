import torch
from torch import nn
import confusion as confusion
from torch import optim
import time
import sklearn.utils as sku
import numpy as np
import os
from IPython.display import display, clear_output
import datautilities as du
import json


class GlitchModel():
    
    def __init__(self, 
                 model : nn.Module, 
                 features : torch.tensor, 
                 labels: torch.tensor, 
                 label_weight_set : dict[int : int] = {0:6,1:1},
                 division_set_ratios : list[float] = [0.8, 0.1, 0.1], 
                 device : str = "cpu"):
        # Some lines of code that allow to set some general parameters of the model
        self.model = model
        self.features = features
        self.labels = labels
        self.learning_rate : float= 0.00001
        self.number_of_epochs : int = 1000
        self.device  = device
        self.train_set_fraction = division_set_ratios[0]
        self.validation_set_fraction = division_set_ratios[1]
        self.test_set_fraction = division_set_ratios[2]
        self.class_weights = None
        self.dataset : torch.TensorDataset = None
        self.label_weight_set = label_weight_set
        self.number_of_classes = len(label_weight_set)
        
        self.training_loss :  np.ndarray = np.array([])
        self.validiation_loss :  np.ndarray = np.array([])
        self.precision :  np.ndarray = np.array([])
        self.recall :  np.ndarray = np.array([])
        self.accuracy :  np.ndarray = np.array([])
        
        self.con_matrix = None
        
        
    
    def setup(self, batchsize = 8192):
        """
        This function prepares the dataset for training, validation, and testing.
        """
        # self.labels = torch.argmax(self.labels, dim=1)
        self.dataset = torch.utils.data.TensorDataset(self.features, self.labels)
        self.train_size = int(self.train_set_fraction * len(self.dataset))
        self.val_size = int(self.validation_set_fraction * len(self.dataset))
        self.test_size = len(self.dataset) - self.train_size - self.val_size
        self.train_set, self.val_set, self.test_set = torch.utils.data.random_split(self.dataset, [self.train_size, self.val_size, self.test_size])
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=batchsize, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(self.val_set, batch_size=batchsize, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=batchsize, shuffle=True)
        classes = np.array(list(self.label_weight_set.keys()))
        class_weights = self.label_weight_set

        temp_class_weights = sku.class_weight.compute_class_weight(class_weight=class_weights, classes=classes, y=self.labels[self.train_set.indices].numpy())
        self.class_weights = torch.tensor(temp_class_weights, dtype=torch.float32, device='cuda')


    def train(self, underfit_mode = False):
        """
        This function implements the training procedure of the model. For all models the ADAM optimizer and CROSSENTROPYLOSS function are used. Several statistics are stored:
        - Training loss
        - Validation loss
        - Precision
        - Recall
        - Accuracy
        - Confusion matrix per epoch
        
        The true class is class 0 in all cases. The batchsize governs the amount of inputs is given to the model at once, before doing the backward pass. A larger batchsize will lead to a faster training, but also the a longer convergence time.
        """
        start_time = time.time()
        criterion = nn.CrossEntropyLoss(self.class_weights)  # Loss function for classification
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1E-5)

        model = self.model.to(self.device)
        loss_list = []
        val_loss_list = []
        precision_list = []
        recall_list = []
        accuracy_list = []
        train_loss_list = []
        con_matrix_list = []

        for epoch in range(self.number_of_epochs):
            model.train()
            train_loss = 0.0

            for batch in self.train_loader:
                inputs, labels = batch
                if underfit_mode== True:
                    # inputs, labels = du.noise_data_reduction(inputs, labels, {0:True, 1 : False, 2 : False, 3: False,4:False, 5:False, 6 :False})
                    noise_mask = labels > 0
                    data_mask = noise_mask == False
                    
                    perm = torch.randperm(data_mask.sum())
                    
                    new_noise_input = inputs[noise_mask][perm]
                    new_noise_label = labels[noise_mask][perm]
                    new_data_input = inputs[data_mask]
                    new_data_label = labels[data_mask]
                    
                    inputs = torch.cat([new_noise_input, new_data_input])
                    labels = torch.cat([new_noise_label, new_data_label])
                    
                
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation phase
            model.eval()
            val_loss = 0.0
            con_matrix = confusion.ConfusionMatrix(self.number_of_classes)

            with torch.no_grad():
                for batch in self.val_loader:
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    clabels = labels.cpu().numpy()
                    cpredicted = predicted.cpu().numpy()
                    con_matrix.add(clabels.astype(int), cpredicted.astype(int))

            train_loss /= len(self.train_loader)
            val_loss /= len(self.val_loader)

            con_matrix.fill()
            recall, precision, accuracy, _ = con_matrix.calculate_properties(true_class = 0)
            con_matrix_list.append(con_matrix.confusion_matrix)
            

            # loss_list.append(train_loss + val_loss)
            val_loss_list.append(val_loss)
            precision_list.append(precision)
            recall_list.append(recall)
            accuracy_list.append(accuracy)
            train_loss_list.append(train_loss)
            clear_output(wait=True)
            display(f"{int((epoch+1)/self.number_of_epochs*100):03}%")
            display(f"Epoch {epoch+1}/{self.number_of_epochs},Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}")

        end_time = time.time()
        print(f"Train time: {end_time - start_time}")
        
        self.model = model
        self.training_loss = np.array(train_loss_list)
        self.validiation_loss = np.array(val_loss_list)
        self.precision = np.array(precision_list)
        self.recall = np.array(recall_list)
        self.accuracy = np.array(accuracy_list)
        self.con_matrix_per_epoch = np.array(con_matrix_list)
        

    def test_model(self):
        """
        This function carries out the general test of the model and gives some final statistics about the model's performance given completely new data.
        """
        self.model.eval()

        correct_tp = 0
        correct_fp = 0
        correct_fn = 0
        correct_tn = 0
        test_precision = 0
        test_recall = 0
        self.con_matrix = confusion.ConfusionMatrix(self.number_of_classes)

        for batch in self.test_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs, 1)
            clabels = labels.cpu().numpy()
            cpredicted = predicted.cpu().numpy()
            self.con_matrix.add(clabels, cpredicted)
            # print()

        
        self.con_matrix.fill()
        test_recall, test_precision, test_accuracy, _ = self.con_matrix.calculate_properties(true_class = 0)
       
        self.test_recall    = test_recall   
        self.test_precision = test_precision
        self.test_accuracy  = test_accuracy 
        
    def save_model(self, path, name):
        """
        This function saves all model settings and the weights of the neural network. It also stores the statistics for visualization and analysis. JSON format is used because of its simplicity and dictionary structure.
        """
        torch.save(self.model.state_dict(), os.path.join(path, name))
        
        settings_name = "model.json"
        
        settings_dict = {
            "learning_rate" : self.learning_rate,
            "number_of_epochs" : self.number_of_epochs,
            "train_fraction_size" : self.train_set_fraction,
            "validation_fraction_size" : self.validation_set_fraction,
            "test_fraction_size" : self.test_set_fraction,
            "class_weights" : self.class_weights.tolist(),
            "label_weights" : self.label_weight_set,
            "number_of_classes" : self.number_of_classes,
            "network_class_name" : self.model.__class__.__name__
        }
        
        
        results_dict = {
            "training_loss" : self.training_loss.tolist(),
            "validation_loss" : self.validiation_loss.tolist(),
            "confusion_matrix_per_epoch" : self.con_matrix_per_epoch.tolist(),
            "confusion_matrix_test" : self.con_matrix.confusion_matrix.tolist(),
            "accuracy_per_epoch" : self.accuracy.tolist(),
            "accuracy_test" : self.test_accuracy,
            "precision_per_epoch" : self.precision.tolist(),
            "precision_test" : self.test_precision,
            "recall_per_epoch" : self.recall.tolist(),
            "recall_test" : self.test_recall,
        }
        
        with open(os.path.join(path, "model_settings.json"), "w") as file:
            json.dump(settings_dict, file, indent=4, separators=(',', ': '))
            
        with open(os.path.join(path, "model_results.json"), "w") as file:
            json.dump(results_dict, file, indent=4, separators=(',', ': '))
        
        
    def read_result_data(self, directory: str):
        """
        Reads the settings and results data from JSON files in the specified directory
        and populates the class attributes.
        """
        settings_dict = None
        results_dict = None

        # Read the model settings
        with open(os.path.join(directory, "model_settings.json"), "r") as file:
            settings_dict = json.load(file)

        # Read the model results
        with open(os.path.join(directory, "model_results.json"), "r") as file:
            results_dict = json.load(file)

        # Populate class attributes from settings_dict
        self.learning_rate = settings_dict["learning_rate"]
        self.number_of_epochs = settings_dict["number_of_epochs"]
        self.train_set_fraction = settings_dict["train_fraction_size"]
        self.validation_set_fraction = settings_dict["validation_fraction_size"]
        self.test_set_fraction = settings_dict["test_fraction_size"]
        self.class_weights = torch.tensor(settings_dict["class_weights"], dtype=torch.float32, device=self.device)
        self.label_weight_set = settings_dict["label_weights"]
        self.number_of_classes = settings_dict["number_of_classes"]

        # Populate class attributes from results_dict
        self.training_loss = np.array(results_dict["training_loss"])
        self.validiation_loss = np.array(results_dict["validation_loss"])
        self.con_matrix_per_epoch = np.array(results_dict["confusion_matrix_per_epoch"])
        self.accuracy = np.array(results_dict["accuracy_per_epoch"])
        self.test_accuracy = results_dict["accuracy_test"]
        self.precision = np.array(results_dict["precision_per_epoch"])
        self.test_precision = results_dict["precision_test"]
        self.recall = np.array(results_dict["recall_per_epoch"])
        self.test_recall = results_dict["recall_test"]
        self.con_matrix = confusion.ConfusionMatrix(self.number_of_classes)
        self.con_matrix.confusion_matrix = np.array(results_dict["confusion_matrix_test"])
        

        # Optional: Log or print information if needed
        print("Model settings and results successfully loaded.")
            
        
