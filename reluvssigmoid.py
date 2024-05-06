"""
compare relu and sigmoid activation functions
"""

# Import the libraries we need for this lab
import torch
from torch import nn
from torchvision import transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
import matplotlib.pylab as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np

torch.manual_seed(2)

# Create the model class using Sigmoid as the activation function
print("creating the model cleass using sigmoid")
class Net(nn.Module):
    """
    Creates a two layer neural network
    """
    def __init__(self, D_in, H1, H2, D_out):
        """
        creates net structure
        """
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)
    
    def forward(self, x):
        """
        creates the x value and sigmoid activation network
        """
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x = self.linear3(x)
        print("sigmoid function created")
        return x

# Create the model class using Relu as the activation function
class NetRelu(nn.Module):
    """
    Creates Model class using Relu  activation
    """
    def __init__(self, D_in, H1, H2, D_out):
        """
        constructs 2 network structure
        """
        super(NetRelu, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)
    
    def forward(self, x):
        """
        defines x and it's order of operations through network
        """
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        print("defined relu created")
        return x
# Model Training Function

def train(model, criterion, train_loader, validation_loader, optimizer, epochs=100):
    i = 0
    print("training model ")
    useful_stuff = {'training_loss': [], 'validation_accuracy': []}  
    # Number of times we train on the entire training dataset
    for epoch in range(epochs):
        # For each batch in the train loader
        for i, (x, y) in enumerate(train_loader):
            # Resets the calculated gradient value, this must be done each time as it accumulates if we do not reset
            print("epoc ",i)
            optimizer.zero_grad()
            # Makes a prediction on the image tensor by flattening it to a 1 by 28*28 tensor
            z = model(x.view(-1, 28 * 28)) 
            # Calculate the loss between the prediction and actual class
            loss = criterion(z, y)
            # Calculates the gradient value with respect to each weight and bias
            loss.backward()
            # Updates the weight and bias according to calculated gradient value
            optimizer.step()
            # Saves the loss
            useful_stuff['training_loss'].append(loss.data.item())
        
        # Counter to keep track of correct predictions
        correct = 0
        # For each batch in the validation dataset
        for x, y in validation_loader:
            # Make a prediction
            z = model(x.view(-1, 28 * 28))
            # Get the class that has the maximum value
            _, label = torch.max(z, 1)
            # Check if our prediction matches the actual class
            correct += (label == y).sum().item()
    
        # Saves the percent accuracy
        accuracy = 100 * (correct / len(validation_dataset))
        useful_stuff['validation_accuracy'].append(accuracy)
    
    return useful_stuff

#MAKE SOME DATA

# Create the training dataset

train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
print("data loaded ",train_dataset)
# Create the validating dataset

validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
print("validation set loaded ",validation_dataset)
# Create the criterion function

criterion = nn.CrossEntropyLoss()

# Create the training data loader and validation data loader object
print("training loader and validation objects ")
# Batch size is 2000 and shuffle=True means the data will be shuffled at every epoch
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2000, shuffle=True)
# Batch size is 5000 and the data will not be shuffled at every epoch
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000, shuffle=False)
print("batch is 5000")
# Set the parameters to create the model
#model with 100 neurons
input_dim = 28 * 28 # Dimension of an image
hidden_dim1 = 50
hidden_dim2 = 50
output_dim = 10 # Number of classes

# Set the number of iterations
# try 35 but it will take long
cust_epochs = 10
print("epochs ",cust_epochs)
##########################################
#TEST RELU AND SIGMOID ACTIVATION DIFFERENCES
##########################################

# Train the model with sigmoid function
print("Testing relu and sigmoid activation")
learning_rate = 0.01
# Create an instance of the Net model
model = Net(input_dim, hidden_dim1, hidden_dim2, output_dim)
print("training sigmoid")
# Create an optimizer that updates model parameters using the learning rate and gradient
print("create an optimizer that updates model parameters using learning",learning_rate, "and gradient")
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# Train the model
training_results = train(model, criterion, train_loader, validation_loader, optimizer, epochs=cust_epochs)

print("training results ",training_results)
# Train the model with relu function
print("training relu")
learning_rate = 0.01
print("learning rate ",learning_rate)
# Create an instance of the NetRelu model
modelRelu = NetRelu(input_dim, hidden_dim1, hidden_dim2, output_dim)
print("modeled relu ")
# Create an optimizer that updates model parameters using the learning rate and gradient
optimizer = torch.optim.SGD(modelRelu.parameters(), lr=learning_rate)
print("relu optimizer ",optimizer)
# Train the model
training_results_relu = train(modelRelu, criterion, train_loader, validation_loader, optimizer, epochs=cust_epochs)

#ANALYZE RESULTS
# Compare the training loss

plt.plot(training_results['training_loss'], label='sigmoid')
plt.plot(training_results_relu['training_loss'], label='relu')
plt.ylabel('loss')
plt.title('training loss iterations')
plt.legend()
plt.show()

# Compare the validation loss

plt.plot(training_results['validation_accuracy'], label = 'sigmoid')
plt.plot(training_results_relu['validation_accuracy'], label = 'relu') 
plt.ylabel('validation accuracy')
plt.xlabel('Iteration')   
plt.legend()
plt.show()

print("~~~~~~~~~~~End of Line~~~~~~~~~~~")