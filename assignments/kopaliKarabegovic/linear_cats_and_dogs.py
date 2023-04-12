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

training_data = PetsDataset("../cifar-10-batches-py", Subset.TRAINING)
test_data = PetsDataset("../cifar-10-batches-py", Subset.TEST)
validation_data = PetsDataset("../cifar-10-batches-py", Subset.VALIDATION)

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
best_acc_val = 0
for epoch in range(epochs):
    print(f"epoch: {epoch+1}")
    iter = train_batches.__iter__()
    for batch_number,batch in enumerate(iter):
        #print(f"epoch: {epoch} - batch:{batch_number}")
        model.train()
        data = batch.data
        labels = batch.label
        data = torch.from_numpy(data)
        labels = torch.from_numpy(labels)
        #forward
        scores = model(data)
        loss = criterion(scores,labels)
        #print(f"loss: {round(loss.item(),2)}")
        #backward
        optimizer.zero_grad() #grad is zero for each batch
        loss.backward()
        #gradient descent, updateing the weights
        optimizer.step()

        #model eval for training data
    model.eval()

    num_correct = 0
    num_samples = 0
    model.eval()
    best_acc = 0
    iter = validation_batches.__iter__()
    with torch.no_grad():
        for batch in iter:
            data = batch.data
            labels = batch.label
            data = torch.from_numpy(data)
            labels = torch.from_numpy(labels)

            scores = model(data)
            _, predictions = scores.max(1)
            num_correct += (predictions == labels).sum()
            num_samples += predictions.size(0)
            acc = float(num_correct) / float(num_samples) * 100
            if acc > best_acc:
                best_acc = acc
        print(f"validation accuracy is: {round(acc, 2)}%")
print(f"best validation acc is: {round(best_acc,2)}%")


def check_accuracy(loader,model):
    num_correct = 0
    num_samples = 0
    best_acc = 0
    model.eval()
    iter = loader.__iter__()
    with torch.no_grad():
        for batch in iter:
            data = batch.data
            labels = batch.label
            data = torch.from_numpy(data)
            labels = torch.from_numpy(labels)

            scores = model(data)
            _, predictions = scores.max(1)
            num_correct += (predictions == labels).sum()
            num_samples += predictions.size(0)
            acc = float(num_correct)/float(num_samples)*100
            if acc > best_acc:
                best_acc = acc
            print(f"Accuracy is: {round(acc,2)}%")
    print(f"best test acc is: {round(best_acc,2)}%")
"""
print("Train accuracy: ")
check_accuracy(train_batches,model)

print("validation accuracy: ")
check_accuracy(validation_batches,model)"""

print("Test accuracy is: ")
check_accuracy(test_batches,model)