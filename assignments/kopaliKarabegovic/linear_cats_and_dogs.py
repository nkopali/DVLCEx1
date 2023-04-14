from visualization import Visualization
from dlvc.datasets.pets import PetsDataset
from dlvc.dataset import Subset
from dlvc.batches import BatchGenerator
from dlvc.test import Accuracy
import dlvc.ops as ops
import cv2
import pickle
import numpy as np
import torch


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
validation_data = PetsDataset(
    "../cifar-10-batches-py", Subset.VALIDATION)

num_of_samples_per_batch = 200
train_batches = BatchGenerator(
    training_data, num_of_samples_per_batch, False, op)
test_batches = BatchGenerator(test_data, num_of_samples_per_batch, False, op)
validation_batches = BatchGenerator(
    validation_data, num_of_samples_per_batch, False, op)

# model for training
model = LinearClassifier(3072, training_data.num_classes())

# parameters
learning_rate = 0.0001
epochs = 100
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("Start of the training")
best_acc_val = Accuracy()
acc_val = Accuracy()
loss_list = []
acc_list = []
for epoch in range(epochs):

    print(f"epoch {epoch+1}")
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

    print(f"train loss: {total_loss:.4f}")
    loss_list.append(total_loss)

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
        print(f"val acc: {acc_val.__str__()}")
        acc_list.append(acc_val.__str__())

        if best_acc_val.__lt__(acc_val):
            best_acc_val._correct = acc_val._correct
            best_acc_val._total = acc_val._total

vis = Visualization(range(epochs), loss_list, "Training Loss", "Loss")
vis.plot()

# vis = Visualization(range(epochs), acc_list, "Validation Accuracy", "Accuracy")
# vis.plot()

print("-" * 50)
print(f"val acc (best): {best_acc_val.__str__()}")

# Model evaluation on test data
model.eval()
test_acc = Accuracy()
iter = test_batches.__iter__()
with torch.no_grad():
    for batch in iter:
        data = batch.data
        labels = batch.label
        data = torch.from_numpy(data)
        labels = torch.from_numpy(labels)

        scores = model(data)
        test_acc.update(scores.detach().numpy(), labels.numpy())

print(f"test acc: {test_acc.__str__()}")

print("End of the training")
