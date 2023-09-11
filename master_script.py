import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import config
import matteos_module as matteo # LP: humble module name! :D

device = torch.device("cuda") if torch.cuda.is_available else torch.device("CPU")
print(f"Device is: {device}")

training_dataset, test_dataset = matteo.download_mnist_data()

training_dataset = matteo.preprocess_data(dataset=training_dataset,
                                          transforms_pool=config.preprocessing["transforms"],
                                          subset_size=config.preprocessing["transformed_size"])

dataloaders = matteo.train_validation_split(dataset=training_dataset,
                                            training_size=config.training_and_validation["training_size"])


hyperparameters = {
                  "dropout_p": config.model_selection["dropout_p"],
                  "learning_rate": config.model_selection["learning_rate"],
                  "weight_decay": config.model_selection["weight_decay"]
                  }  
hyperparameter_combinations = matteo.combine(hyperparameters)

optimal_hyperparameters = matteo.hyperparameter_tuning(combinations=hyperparameter_combinations,
                                                       device=device,
                                                       dataloaders=dataloaders)

torch.manual_seed(0)
model = matteo.CNN(dropout_p=optimal_hyperparameters.dropout_p)
model.to(device)
losses = matteo.train_and_validate(model=model,
                                   device=device,
                                   combination=optimal_hyperparameters,
                                   epochs=config.training_and_validation["epochs"],
                                   dataloaders=dataloaders)

matteo.plot_losses(size=(12,8),
                   losses=losses,
                   labels=["Training loss", "Validation loss"])

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=len(test_dataset),
                         shuffle=False,
                         num_workers=0)

predicted_image_labels = matteo.test(model=model,
                                     device=device,
                                     dataloader=test_loader)

true_image_labels = test_dataset.targets.cpu().numpy()
predicted_image_labels = torch.argmax(input=predicted_image_labels,dim=1).cpu().numpy()
confusion_matrix = matteo.plot_confusion_matrix(true=true_image_labels,
                                                predicted=predicted_image_labels,
                                                classes=10)

matteo.plot_incorrect(dataset=test_dataset,
                      confusion_matrix=confusion_matrix,
                      classes=10)



