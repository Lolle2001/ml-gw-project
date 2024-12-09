import torch
from torch import nn
from torch import optim


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





def train_model(model : nn.Module,
                train_loader : torch.tensor, 
                val_loader  : torch.tensor, 
                epochs : int =50, 
                lr : float =0.001, 
                device : str='cpu'):
    
    criterion = nn.CrossEntropyLoss()  # Loss function for classification
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model = model.to(device)
    loss_list=[]
    val_loss_list=[]
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
        correct = 0
        total = 0
        

        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        accuracy = correct / total

        
        loss_list.append(loss)
        val_loss_list.append(val_loss)
        accuracy_list.append(accuracy)
        train_loss_list.append(train_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")

    return model, loss_list,val_loss_list,accuracy_list,train_loss_list 