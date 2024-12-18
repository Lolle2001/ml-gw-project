import torch
from torch import nn
import confusion as con
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
            nn.Linear(hidden_dim, output_dim),
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
            nn.ReLU(),
            nn.Linear(8, 10),
            nn.ReLU(),
            nn.Linear(10, output_dim),
            nn.Softmax(dim=1)
            # nn.Softmax(dim=1)  # Softmax for multi-class classification
        )

    def forward(self, x):
        return self.network(x)






def train_model(model : nn.Module,
                train_loader : torch.tensor, 
                val_loader  : torch.tensor, 
                epochs : int =50, 
                lr : float =0.001,
                weights : torch.tensor = None, 
                device : str='cpu'):
    starttime = time.time()
    criterion = nn.CrossEntropyLoss(weights)  # Loss function for classification
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model = model.to(device)
    loss_list=[]
    val_loss_list=[]
    precision_list=[]
    recall_list=[]
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
        # accuracy = 0
        correct_tp = 0
        correct_fp = 0
        correct_fn = 0
        correct_tn = 0
        precision = 0
        recall = 0
        con_matrix = con.ConfusionMatrix(2)
        # print(con_matrix.confusion_matrix.shape)

        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                clabels = labels.cpu().numpy()
                cpredicted = predicted.cpu().numpy()
                con_matrix.add(clabels.astype(int), cpredicted.astype(int))
                
              

                # correct += (predicted == labels).sum().item()
                # total += labels.size(0)

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

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

        print("Epoch {0}/{1}, Train Loss: {2:.4f}, Val Loss: {3:.4f}, Precision: {4:.4f}, Recall: {5:.4f}, Accuracy: {6:.4f}".format(
                  (epoch+1),epochs,
                  train_loss,
                    val_loss,
                    precision,
                    recall,
                    accuracy
              ))


    enddtime = time.time()
    print(f"Train time: {enddtime - starttime}")
    return model, loss_list,train_loss_list,val_loss_list,precision_list,recall_list, accuracy_list