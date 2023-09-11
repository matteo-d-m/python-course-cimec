from itertools import product
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision
import torchvision.transforms as transforms

from sklearn.metrics import confusion_matrix as sklearn_confusion

import config

def download_mnist_data():
  training_dataset = torchvision.datasets.MNIST(root='classifier data', 
                                                train=True, 
                                                download=True, 
                                                transform=transforms.ToTensor())
  test_dataset = torchvision.datasets.MNIST(root='classifier data', 
                                            train=False, 
                                            download=True, 
                                            transform=transforms.ToTensor())
  return training_dataset, test_dataset

class TransformedData(Dataset):
  def __init__(self, list_of_labelled_images):
    self.list_of_labelled_images = list_of_labelled_images

  def __len__(self):
    return len(self.list_of_labelled_images)

  def __getitem__(self, idx):
    sample = (self.list_of_labelled_images[idx])
    image = sample[0]
    label = sample[1]
    return sample
  
def preprocess_data(dataset, transforms_pool, subset_size):
  """Augments data through random affine transformations.
  
  Parameters:
  dataset -- a PyTorch dataset
  transforms_pool -- the affine transforms to use (list)
  subset_size -- fraction of data to transform (float)
  
  Returns:
  dataset -- the original PyTorch dataset, plus a 'subset_size'
  percentage of transformed samples
  """
  sample_and_apply = transforms.RandomApply(torch.nn.ModuleList(transforms_pool), p=1)
  idxs_to_copy_from = np.random.randint(low=0, 
                                        high=len(dataset),
                                        size=(int(len(dataset)*subset_size))).tolist()
  copied_data = [dataset[idx] for idx in idxs_to_copy_from]
  transformed_copies = [(sample_and_apply(copy[0]), copy[1]) for copy in copied_data]
  transformed_fraction = TransformedData(transformed_copies)
  dataset = torch.utils.data.ConcatDataset([dataset, transformed_fraction])
  return dataset

def train_validation_split(dataset, training_size):
  """Splits a PyTorch dataset in two unequally large subsets, then 
  instantiates one dataloader each.
  
  Parameters:
  dataset -- a PyTorch dataset
  training_size -- fraction of data to use for training (float)

  Returns:
  training_loader -- PyTorch dataloader for training data
  validation_loader -- PyTorch dataloader for validation data
  """

  training = int(len(dataset) * training_size)
  training_data, validation_data = random_split(dataset=dataset, 
                                                lengths=[training, len(dataset)-training],
                                                generator=torch.Generator().manual_seed(0))
  training_loader = DataLoader(dataset=training_data,
                               batch_size=config.training_and_validation["batch_size"],
                               shuffle=True,
                               num_workers=2,
                               pin_memory=True)
  validation_loader = DataLoader(dataset=validation_data,
                                 batch_size=len(validation_data),
                                 shuffle=False,
                                 num_workers=2,
                                 pin_memory=True)
  return training_loader, validation_loader
  
class CNN(nn.Module):
  """A convolutional neural network. Check PyTorch docs"""

  def __init__(self, dropout_p):                                              
    super().__init__() 
    self.conv1 = nn.Conv2d(in_channels=config.model["in_channels_first"], 
                           out_channels=config.model["out_channels_first"], 
                           kernel_size=config.model["kernel_size_first"], 
                           stride=config.model["stride_first"])
    self.conv2 = nn.Conv2d(in_channels=config.model["in_channels_second"],
                           out_channels=config.model["out_channels_second"],
                           kernel_size=config.model["kernel_size_second"],
                           stride=config.model["stride_second"])
    self.flatten = nn.Flatten(start_dim=1) 
    self.fc1 = nn.Linear(in_features=config.model["in_features_first"], 
                         out_features=config.model["out_features_first"]) 
    self.fc2 = nn.Linear(in_features=config.model["in_features_second"],
                         out_features=config.model["out_features_second"])
    self.drop = nn.Dropout(p=dropout_p)
    self.act = nn.ReLU()

  def forward(self, x):
    x = self.act(self.conv1(x))
    x = self.act(self.conv2(x))
    x = self.flatten(x)        
    x = self.act(self.fc1(x))
    x = self.act(self.drop(x))
    out = self.fc2(x)
    return out
    
def train_and_validate(model, device, combination, epochs, dataloaders):
  """Performs model training and validation.
  
  Parameters:
  model -- a PyTorch model instance
  device -- where to run computations (torch device object)
  combination -- a combination of hyperparameter values (namedtuple)
  epochs -- number of model runs (int)
  dataloaders -- PyTorch dataloader instances (tuple)
  """

  cross_entropy_loss = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), 
                         lr = combination.learning_rate,
                         weight_decay = combination.weight_decay)
  training_loss_log = []
  validation_loss_log = []

  # LP: since you like being fancy, for the future I recommend the tqdm library for progress bars!
  for epoch in range(epochs):
    training_loss = []
    model.train()
    for batch in dataloaders[0]:
      image = batch[0].to(device)
      label = batch[1].to(device)
      output = model(image)
      loss = cross_entropy_loss(output, label)
      model.zero_grad()
      loss.backward()
      optimizer.step()
      loss = loss.detach().cpu().numpy()
      training_loss.append(loss)
    validation_loss = []
    model.eval()
    with torch.no_grad():
      for batch in dataloaders[1]:
        image = batch[0].to(device)
        label = batch[1].to(device)
        output = model(image)
        loss = cross_entropy_loss(output, label)
        loss = loss.detach().cpu().numpy()
        validation_loss.append(loss)
    training_loss = np.mean(training_loss)
    training_loss_log.append(training_loss)
    validation_loss = np.mean(validation_loss)
    validation_loss_log.append(validation_loss)
    print(f"EPOCH {epoch+1} - TRAINING LOSS: {training_loss: .2f} - VALIDATION LOSS: {validation_loss: .2f}")
    if epoch == epochs-1:
      print("Finished")
  torch.save(model.state_dict(), 'model_parameters.torch')
  return training_loss_log, validation_loss_log

def combine(hyperparameters):
  """Constructs combinations of hyperparameter values.

  Parameters:
  hyperparameters -- map between hyperparameter names and candidate
  values (dict, str:list)

  Returns:
  candidates -- combinations of hyperparameters values (list of namedtuples)
  """

  # LP even namedtuples! superfancy!
  candidate = namedtuple('Candidate', hyperparameters.keys()) 
  candidates = []
  for combination in product(*hyperparameters.values()): 
    candidates.append(candidate(*combination))
  return candidates

def hyperparameter_tuning(combinations, device, dataloaders):
  """Chooses the best combination of hyperparameters.
  
  Parameters:
  combinations -- hyperparameter combinations to evaluate (namedtuple)
  device -- where to run computations (torch device object)
  dataloaders -- PyTorch dataloader instances (tuple)
  """

  scores = []
  for combination in combinations:
    model = CNN(dropout_p=combination.dropout_p)
    model.to(device)
    print(f"Combination {combinations.index(combination)+1} of {len(combinations)}")
    score = train_and_validate(model=model, 
                               device=device, 
                               combination=combination, 
                               epochs=config.model_selection["epochs"], 
                               dataloaders=dataloaders) 
    scores.append(score)
  print("Model selection finished!")
  training_scores = []
  validation_scores = []
  for score in scores:
    training, validation = score
    training_scores.append(training)
    validation_scores.append(validation)
  least_validation_score = min(validation_scores)
  idx = validation_scores.index(least_validation_score)
  winner = combinations[idx]
  return winner

def plot_losses(size, losses, labels):
  """Draws line plots of losses (i.e., model errors) vs. epoch number.
  
  Parameters:
  size -- figsize (tuple)
  losses -- the losses to draw (list)
  labels -- the graph's lables (list)
  """

  # LP: very minor. Stuff that have module configuration effects such as
  # this one should not be used in a function! Otherwise they have side effects
  # (unless the function is specifically deputed to have such effectso)
  # Love the darkstyle though!
  plt.style.use("dark_background")
  plt.figure(figsize=(size[0],size[1]))
  plt.semilogy(losses[0], label=labels[0])
  plt.semilogy(losses[1], label=labels[1])
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend()
  plt.grid()
  plt.show()

def test(model, device, dataloader):
  """Evaluates the model on novel samples.
  
  Parameters:
  model -- a PyTorch model instance
  device -- where to run computations (torch device object)
  dataloader -- a PyTorch dataloader instance
  """

  images = []
  labels = []
  predictions = []
  model.eval()
  with torch.no_grad():
    for sample in dataloader:
      image = sample[0].to(device)
      label = sample[1].to(device)
      pred = model(image)
      images.append(image)
      labels.append(label)
      predictions.append(pred)
  images = torch.cat(images)
  labels = torch.cat(labels)
  predictions = torch.cat(predictions)
  correct = predictions.argmax(dim=1).eq(labels).sum()
  accuracy = correct*100/len(labels)
  print(f"TEST ACCURACY: {accuracy: .2f}%")
  return predictions

def plot_confusion_matrix(true, predicted, classes):
  """Plots a heatmap-style confusion matrix. 
  Leverages scikit-learn's 'confusion_matrix()'

  Parameters:
  true -- ground truth labels (array-like)
  predicted -- labels predicted by the model (array-like)
  classes -- the number of classes in the dataset (int)
  """
  
  matrix = sklearn_confusion(true, predicted)
  plt.figure(figsize=(12,10))
  plt.imshow(matrix, interpolation = 'nearest', cmap ='Reds')
  matrix_cells = product(range(matrix.shape[0]), range(matrix.shape[1]))
  for row_index, column_index in matrix_cells:
    plt.text(x=row_index, 
             y=column_index, 
             s=matrix[row_index][column_index],
             horizontalalignment="center",
             verticalalignment="center",
             color="white" if row_index == column_index else "black")
  ticks = np.arange(classes)
  plt.xticks(ticks)
  plt.yticks(ticks)
  plt.xlabel("Predicted label")
  plt.ylabel("True label")
  plt.title("Test confusion matrix")
  plt.colorbar()
  return matrix

def plot_incorrect(dataset, confusion_matrix, classes):
  """Creates a bar chart of test mistakes per class.
  
  Parameters:
  dataset -- a PyTorch dataset
  confusion_matrix -- a confusion matrix (2darray)
  classes -- the number of classes in the dataset (int)
  """

  bins = dataset.targets.bincount()
  incorrect = [bins[i] - confusion_matrix[i][i] for i in range(len(bins))] 
  bars = np.arange(classes) 
  plt.figure(figsize=(12,8))
  plt.bar(bars, incorrect)
  plt.xticks(bars)
  plt.ylabel("Incorrectly classified")
  plt.title("Number of mistakes per class")
  plt.grid(axis="y")
  plt.show()

def visualize_filter(layer_filters, filter_index, reshape_dims):
  """Plots filter (i.e., kernel) values on a grayscale.
  
  Parameters:
  layer_filters -- a PyTorch tensor to index into
  filter_index -- the index of the desired filter (int)
  reshape_dims -- list of output dimensions for the filters tensor
  """

  filter = layer_filters[:,filter_index,:,:]
  filter = filter.reshape(reshape_dims[0],
                          reshape_dims[1],
                          reshape_dims[2],
                          reshape_dims[3]) # 4 5 5 5
  _, axs = plt.subplots(4,5,figsize=(12,8))
  for i in range(filter.shape[0]):
    for j in range(filter.shape[1]):
      axs[i][j].imshow(filter[i][j], cmap="gray") 
      axs[i][j].set_xticks([])
      axs[i][j].set_yticks([])
  plt.show()
