import torch
from torch import nn
from data_setup import create_dataloaders
import model_setup
from utils import save_model

# check for device availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# data loaders
train_loader, test_loader = create_dataloaders()

# instantiate the model
model = model_setup.ANN().to(device)

# define the loss function
loss_fn = nn.CrossEntropyLoss()

# setup optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# start the training loop
epochs = 5

for epoch in range(epochs):
  model.train()

  running_loss = 0.0
  for images, labels in train_loader:
    # move images and labels to correct device
    images, labels = images.to(device), labels.to(device)

    # 1. Forward pass
    y_pred = model(images)

    # 2. Calculate loss
    loss = loss_fn(y_pred, labels)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Optimizer step
    optimizer.step()
    running_loss += loss.item()

  print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

# check test set accuracy
correct = 0
total = 0
model.eval()
with torch.inference_mode():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Test Accuracy: {100 * correct / total:.2f}%")

save_model(model=model,
           target_dir='',
           model_name='simple_neural_network.pth')