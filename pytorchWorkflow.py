import torch
from torch import nn  # contains all pytorch's building blocks for neural networks
from torch import optim
from torchmetrics import Accuracy
import matplotlib.pyplot as plt
from pathlib import Path

# 1.Data preparation
# Splitting data: training and test sets
# training set, validation set, test set
start = 0
end = 1
step = 0.02
weights = 3
bias = 7
# get the data
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weights*X+bias
# split data
train_split = int(0.8*len(X))  # using 80% of the set for training
X_train, y_train = X[:train_split], y[:train_split]  # using 80% for training
X_test, y_test = X[train_split:], y[train_split:]  # using 20% for testing


# 2.Build the model
# set the device, agonostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# opt. create an accuracy function
torchmetric_accuracy = Accuracy().to(device)


class LinearRegressionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1,
                                      out_features=1)
    # forward method to define the computation in the model
    # any subclass of nn.Module needs to override the forward

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     return self.weights * x + self.bias
    #     # this is the linear regression formula
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


# make predictions with the model
model_0 = LinearRegressionModel()
# with torch.inference_mode():  # turns off the gradient tracking, makes the code faster


# 3.Training the model
# setup a loss function
loss_fn = nn.L1Loss()
# setup an optimizer
optimizer = optim.SGD(params=model_0.parameters(),
                      lr=0.01)  # lr=learning rate, a hyperparameter
# create a training loop
epochs = 5  # hyperparameter
for epoch in range(epochs):
    # set the model to training mode
    model_0.train()
    # 1. forward pass
    y_pred = model_0(X_train)
    # 2. calculate the loss
    loss = loss_fn(y_pred, y_train)  # (input, target)
    # 3.optimizer zero grad
    optimizer.zero_grad()
    # 4.backpropagation on the loss
    loss.backward()
    # 5.step the optimizer(gradient descent)
    optimizer.step()
    # test the accuracy


# 4. TESTING, evaluating
# you need to create a testing loop, I haven't made one here
model_0.eval()  # turn off the gradient
with torch.inference_mode():
    test_pred = model_0(X_test)
    test_loss = loss_fn(test_pred, y_train)
if epoch % 10:
    print(f"Epoch : {epoch} | Loss: {loss} | Test_loss: {test_loss}")
    print(model_0.state_dict())

# 5. SAVING A MODEL
# three methods torch.save(), torch.load(), torch.nn.Module.load_state_dict()
# a.create a model directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
# b.create the model save path
MODEL_NAME = "01_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
# save the model
print(f"Saving the model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(),
           f=MODEL_SAVE_PATH)

# 6. LOADING A MODEL
loaded_model_0 = LinearRegressionModel()
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
