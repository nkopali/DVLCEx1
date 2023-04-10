from dlvc.datasets.pets import PetsDataset
from dlvc.dataset import Subset
from dlvc.batches import BatchGenerator
from dlvc.test import Accuracy
import dlvc.ops as ops
import cv2, pickle, numpy as np, torch


# TODO: Define the network architecture of your linear classifier.
class LinearClassifier(torch.nn.Module):
  def __init__(self, input_dim, num_classes):
      super(LinearClassifier, self).__init__()
      self.fc = torch.nn.Linear(input_dim, num_classes)

  def forward(self, x):
      x = self.fc(x)
      return x

op = ops.chain([
    ops.vectorize(),
    ops.type_cast(np.float32),
    ops.add(-127.5),
    ops.mul(1/127.5),
])

training_data = PetsDataset("../cifar-10-batches-py", Subset.TRAINING)
test_data = PetsDataset("../cifar-10-batches-py", Subset.TEST)
validation_data = PetsDataset("../cifar-10-batches-py", Subset.VALIDATION)


# Create a BatchGenerator for each dataset using the input transformation chain op.
training_batch = BatchGenerator(training_data, len(training_data), shuffle=True, op=op)
validation_batch = BatchGenerator(validation_data, len(validation_data), shuffle=False, op=op)
test_batch = BatchGenerator(test_data, len(test_data), shuffle=False, op=op)

# TODO: Create the LinearClassifier, loss function and optimizer. 


'''
TODO: Train a model for multiple epochs, measure the classification accuracy on the validation dataset throughout the training and save the best performing model. 
After training, measure the classification accuracy of the best perfroming model on the test dataset. Document your findings in the report.
'''

# Define the LinearClassifier, loss function and optimizer.
model = LinearClassifier(input_dim=3072, num_classes=2)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Define variables for best accuracy and corresponding model weights
best_accuracy = 0.0
best_model_weights = None

# Define number of epochs and iterate through training data for each epoch
num_epochs = 10
for epoch in range(num_epochs):

    # Train the model on training data
    for x, y_true in training_batch:
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        optimizer.step()

    # Evaluate the model on validation data and measure accuracy
    accuracy = Accuracy()
    with torch.no_grad():
        for x, y_true in validation_batch:
            y_pred = model(x)
            y_pred = torch.argmax(y_pred, dim=1)
            accuracy.update_state(y_true.numpy(), y_pred.numpy())

    # Check if current accuracy is better than previous best accuracy and save model weights if it is
    if accuracy.get() > best_accuracy:
        best_accuracy = accuracy.get()
        best_model_weights = model.state_dict()

    # Print current epoch and validation accuracy
    print(f"Epoch {epoch+1}, Validation Accuracy: {accuracy.get()}")

# Load best performing model weights and evaluate on test data
model.load_state_dict(best_model_weights)
accuracy = Accuracy()
with torch.no_grad():
    for x, y_true in test_batch:
        y_pred = model(x)
        y_pred = torch.argmax(y_pred, dim=1)
        accuracy.update_state(y_true.numpy(), y_pred.numpy())

# Print test accuracy
print(f"Test Accuracy: {accuracy.get()}")