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
      self.softmax = torch.nn.Softmax(dim=1)

  def forward(self, x):
      x = self.fc(x)
      return x

op = ops.chain([
    ops.vectorize(),
    ops.type_cast(np.float32),
    ops.add(-127.5),
    ops.mul(1/127.5),
])

training_data = PetsDataset("assignments\cifar-10-batches-py", Subset.TRAINING)
test_data = PetsDataset("assignments\cifar-10-batches-py", Subset.TEST)
validation_data = PetsDataset("assignments\cifar-10-batches-py", Subset.VALIDATION)

num_of_samples_per_batch = 100
train_batches = BatchGenerator(training_data,num_of_samples_per_batch, False)
test_batches =  BatchGenerator(test_data,num_of_samples_per_batch, False)
validation_batches = BatchGenerator(validation_data,num_of_samples_per_batch, False)

# TODO: Create the LinearClassifier, loss function and optimizer.

# model for training
model = LinearClassifier(3072,training_data.num_classes())#.to(device)

#parameters
learning_rate = 0.0001
epochs = 100
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)


print("Start of the training")
best_acc_val = Accuracy()
acc_val = Accuracy()
for epoch in range(epochs):

    print(f"epoch: {epoch+1}")
    iter = train_batches.__iter__()

    total_loss = 0
    for batch_number, batch in enumerate(iter):
        model.train()
        data = batch.data
        labels = batch.label
        data = torch.from_numpy(data)
        labels = torch.from_numpy(labels)
        # forward
        scores = model(data)
        loss = criterion(scores, labels)
        total_loss += loss.item()

        # backward
        optimizer.zero_grad()  # grad is zero for each batch
        loss.backward()
        # gradient descent, updating the weights
        optimizer.step()

    print(f"total training loss for epoch {epoch+1}: {total_loss:.4f}")

    # model eval for training data
    model.eval()

    acc_val.reset()
    iter = validation_batches.__iter__()
    with torch.no_grad():
        for batch in iter:
            data = batch.data
            labels = batch.label
            data = torch.from_numpy(data)
            labels = torch.from_numpy(labels)

            scores = model(data)
            acc_val.update(scores.detach().numpy(), labels.numpy())
        print(f"validation accuracy is: {acc_val.__str__()}")

        if best_acc_val.__lt__(acc_val):
            best_acc_val._correct = acc_val._correct
            best_acc_val._total = acc_val._total

print(f"best validation acc is: {best_acc_val.__str__()}")
print("End of the training")