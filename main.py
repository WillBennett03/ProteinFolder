"""
###############
### main.py ###
###############

~ Will Bennett 17/06/2021

This module runs the models with the preprocessed data to allow for training
and predictions to occur

"""

from torch.nn.modules import transformer
import model
import preprocessing
from torch.utils.data import DataLoader
import torch

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == '__main__':
    x,y = preprocessing.getRecord('protein_test')
    transformer = model.Model(20, 6, 6, 6, 6)
    data = list(zip(x,y))
    training_data_length = int(len(data) * 0.8) 
    training_data = data[training_data_length:]
    testing_data = data[:training_data_length]
    train_Dataloader = DataLoader(training_data, shuffle=True)
    test_Dataloader = DataLoader(testing_data, shuffle=True)

    learning_rate = 1e-6
    optimizer = torch.optim.ADAM(transformer.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()
    epoch = 10

    for t in range(epoch):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_Dataloader, model, loss_fn, optimizer)
        test_loop(test_Dataloader, model, loss_fn)
    print("Done!")