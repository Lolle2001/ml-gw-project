import torch
from torch import nn
import confusion as confusion
from torch import optim
import time
import sklearn.utils as sku
import numpy as np


class GlitchModel():
    
    def __init__(self, 
                 model : nn.Module, 
                 features : torch.tensor, 
                 labels: torch.tensor, 
                 label_weight_set : dict[int : int] = {0:6,1:1},
                 division_set_ratios : list[float] = [0.8, 0.1, 0.1], 
                 device : str = "cpu"):
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
        
        self.training_loss :  np.ndarray = np.array([])
        self.validiation_loss :  np.ndarray = np.array([])
        self.precision :  np.ndarray = np.array([])
        self.recall :  np.ndarray = np.array([])
        self.accuracy :  np.ndarray = np.array([])
    
    def setup(self):
        self.dataset = torch.utils.data.TensorDataset(self.features, self.labels)
        self.train_size = int(self.train_set_fraction * len(self.dataset))
        self.val_size = int(self.validation_set_fraction * len(self.dataset))
        self.test_size = len(self.dataset) - self.train_size - self.val_size
        self.train_set, self.val_set, self.test_set = torch.utils.data.random_split(self.dataset, [self.train_size, self.val_size, self.test_size])
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=4128, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(self.val_set, batch_size=4128, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=4128, shuffle=True)
        temp_class_weights = sku.class_weight.compute_class_weight(self.label_weight_set, classes=np.array(self.label_weight_set.keys()), y=self.labels[self.train_set.indices])
        self.class_weights = torch.tensor(temp_class_weights, dtype=torch.float32, device='cuda')


    def train(self):
        start_time = time.time()
        criterion = nn.CrossEntropyLoss(weights)  # Loss function for classification
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        model = self.model.to(self.device)
        loss_list = []
        val_loss_list = []
        precision_list = []
        recall_list = []
        accuracy_list = []
        train_loss_list = []

        for epoch in range(self.number_of_epochs):
            model.train()
            train_loss = 0.0

            for batch in self.train_loader:
                inputs, labels = batch
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
            con_matrix = confusion.ConfusionMatrix(2)

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

            con_matrix.calculate()
            precision = con_matrix.precision
            accuracy = con_matrix.accuracy
            recall = con_matrix.recall

            loss_list.append(train_loss + val_loss)
            val_loss_list.append(val_loss)
            precision_list.append(precision)
            recall_list.append(recall)
            accuracy_list.append(accuracy)
            train_loss_list.append(train_loss)

            print(f"Epoch {epoch+1}/{self.number_of_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}")

        end_time = time.time()
        print(f"Train time: {end_time - start_time}")
        
        self.model = model
        self.training_loss = np.array(train_loss_list)
        self.validiation_loss = np.array(val_loss_list)
        self.precision = np.array(precision_list)
        self.recall = np.array(recall_list)
        self.accuracy = np.array(accuracy_list)
        

        
