import torch
from torch import nn
from torch import optim
import time


class GlitchClassifier(nn.Module):
    def __init__(self, input_dim : int=6, hidden_dim :int=32, output_dim:int=2):
        super(GlitchClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
            # nn.Softmax(dim=1)  # Softmax for multi-class classification
        )

    def forward(self, x):
        return self.network(x)
    

class GlitchClassifierDynamic(nn.Module):
    def __init__(self, input_dim : int=6, hidden_dim :int=32, output_dim:int=2):
        super(GlitchClassifierDynamic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 8),
            nn.ELU(),
            nn.Linear(8, 10),
            nn.ReLU(),
            nn.Linear(10, output_dim),
            nn.Softmax(dim=1),
            nn.Linear(output_dim, output_dim)
            # nn.Softmax(dim=1)  # Softmax for multi-class classification
        )

    def forward(self, x):
        return self.network(x)






def train_model(model : nn.Module,
                train_loader : torch.tensor, 
                val_loader  : torch.tensor, 
                epochs : int =50, 
                lr : float =0.001, 
                device : str='cpu'):
    starttime = time.time()
    criterion = nn.CrossEntropyLoss()  # Loss function for classification
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model = model.to(device)
    loss_list=[]
    val_loss_list=[]
    accuracy_0_list=[]
    accuracy_1_list=[]
    accuracy_list=[]
    train_loss_list = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

       

        for batch in train_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_0 = 0
        correct_1 = 0
        total_0 = 0
        total_1 = 0
        

        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                mask_0 = labels == 0
                mask_1 = labels == 1
                correct_0 += (predicted[mask_0] == labels[mask_0]).sum().item()
                correct_1 += (predicted[mask_1] == labels[mask_1]).sum().item()
                total_0 += labels[mask_0].size(0)
                total_1 += labels[mask_1].size(0)

                # correct += (predicted == labels).sum().item()
                # total += labels.size(0)

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        accuracy_0 = correct_0 / total_0
        accuracy_1 = correct_1 / total_1
        accuracy = (correct_0 + correct_1) / (total_0+total_1)

        
        loss_list.append(train_loss + val_loss)
        val_loss_list.append(val_loss)
        accuracy_0_list.append(accuracy_0)
        accuracy_1_list.append(accuracy_1)
        accuracy_list.append(accuracy)
        train_loss_list.append(train_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy(0): {accuracy_0:.4f}, Accuracy(1): {accuracy_1:.4f}")


    enddtime = time.time()
    print(f"Train time: {enddtime - starttime}")
    return model, loss_list,train_loss_list,val_loss_list,accuracy_list, accuracy_0_list, accuracy_1_list