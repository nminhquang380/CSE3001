from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import tqdm.notebook as tq
import wandb
import torchmetrics
from torch.utils.data import DataLoader

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import json

# Determine which device on import, and then use that elsewhere.
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)


def plot_confusion_matrix(cm, class_names):
    '''
        cm: the confusion matrix that we wish to plot
        class_names: the names of the classes 
    '''

    # this normalizes the confusion matrix
    cm = cm.cpu().detach().numpy()
    cm = cm.astype(np.float32) / cm.sum(axis=1)[:, None]
    
    df_cm = pd.DataFrame(cm, class_names, class_names)
    ax = sn.heatmap(df_cm, annot=True, cmap='flare')

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.show()
    
def count_classes(preds):
    '''
    Counts the number of predictions per class given preds, a tensor
    shaped [batch, n_classes], where the maximum per preds[i]
    is considered the "predicted class" for batch element i.
    '''
    pred_classes = preds.argmax(dim=1)
    n_classes = preds.shape[1]
    return [(pred_classes == c).sum().item() for c in range(n_classes)]

def train_epoch(epoch, model, optimizer, criterion, loader, num_classes, device):
    '''
    Train the model on the entire training set precisely once (one epoch).
    Lab 6 has a very similar function.
    '''

    model.train()

    # Initialize metrics
    # TODO: Task 1b - initialize the following torchmetrics metrics
    #       the average epoch loss per example
    #       accuracy
    #       unweighted average recall
    epoch_loss = torchmetrics.MeanMetric().to(device)
    acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes).to(device)
    recall_metric = torchmetrics.Recall(task='multiclass', num_classes=num_classes, average='macro').to(device)

    for i, (inputs, lbls) in enumerate(loader):
        try:
          inputs, lbls = inputs.to(device), lbls.to(device)
        except:
          lbls = torch.from_numpy(np.array(lbls))
          inputs, lbls = inputs.to(device), lbls.to(device)
        # Update model weights
        # TODO: Task 1b - Use the batch to update the weights of the model
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        # TODO: Task 1b - accumulate each of the 3 metrics you initialized.
        epoch_loss(loss)
        acc_metric(outputs, lbls)
        recall_metric(outputs, lbls)

    # Calculate epoch metrics, and store in a dictionary for wandb
    # TODO Task 1b - compute the three metrics
   
    metrics_dict = {
        'Loss_train': epoch_loss.compute().cpu().detach().numpy().tolist(),
        'Accuracy_train': acc_metric.compute().cpu().detach().numpy().tolist(),
        'UAR_train': recall_metric.compute().cpu().detach().numpy().tolist(),
    }

    return metrics_dict

def val_epoch(epoch, model, criterion, loader, num_classes, device):
    '''
    Evaluate the model on the entire validation set.
    '''
    model.eval()
    
    # Initialize metrics
    # TODO: Task 1b - initialize the following torchmetrics metrics
    #       the average epoch loss per example
    #       accuracy
    #       unweighted average recall
    epoch_loss = torchmetrics.MeanMetric().to(device)
    acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes).to(device)
    recall_metric = torchmetrics.Recall(task='multiclass', num_classes=num_classes, average='macro').to(device)

    
    # TODO: Task 1c - initialize a confusion matrix torchmetrics object
    confusion_matrix = torchmetrics.classification.ConfusionMatrix(task='multiclass', num_classes=num_classes).to(device)
    
    for inputs, lbls in loader:
        inputs, lbls = inputs.to(device), lbls.to(device)

        # TODO Task 1b - Obtain validation loss (use torch.no_grad())
        with torch.no_grad():
          outputs = model(inputs)
          loss = criterion(outputs, lbls)

        # Accumulate metrics
        # TODO: Task 1b - accumulate each of the 3 metrics you initialized
        #       Look at trainer.py of lab 6 for inspiration.
        #       This will take the loss, outputs and lbls from this batch
        #       to update each of the metric object's internal state. 
        epoch_loss(loss)
        acc_metric(outputs, lbls)
        recall_metric(outputs, lbls)
        
        # TODO: Task 1c - acculmate confusion matrix
        confusion_matrix(outputs, lbls)

         
    # Calculate epoch metrics, and store in a dictionary for wandb
    # TODO Task 1b - compute the three metrics 
    metrics_dict = {
        'Loss_val': epoch_loss.compute().cpu().detach().numpy().tolist(),
        'Accuracy_val': acc_metric.compute().cpu().detach().numpy().tolist(),
        'UAR_val': recall_metric.compute().cpu().detach().numpy().tolist(),
    }

    # Compute the confusion matrix
    # TODO: Task 1c - compute the confusion matrix and store it in cm
    cm = confusion_matrix.compute()

    return metrics_dict, cm
    


def train_model(model, train_loader, val_loader, optimizer, criterion,
                class_names, n_epochs, project_name, ident_str=None):
                
    num_classes = len(class_names)
    model.to(device)

    # Initialise Weights and Biases (wandb) project
    # if ident_str is None:
    ident_str += datetime.now().strftime("  %Y%m%d_%H%M%S")
    exp_name = f"{model.__class__.__name__}_{ident_str}"
    run = wandb.init(project=project_name, name=exp_name)
    model_info = {
      'exp_name': exp_name,
      'model_architect': str(model.cuda()),
    }
    # Specify the file path for the JSON file
    models_json_file = '/content/drive/MyDrive/Uni/CSE3001/Assignment/saved_models/models.json'

    # Write the model information to the JSON file
    with open(models_json_file, "a") as json_file:
      json.dump(model_info, json_file, indent=4)

    try:
        # Train by iterating over epochs
        for epoch in tq.tqdm(range(n_epochs), total=n_epochs, desc='Epochs'):
            train_metrics_dict = train_epoch(epoch, model, optimizer, criterion,
                    train_loader, num_classes, device)
                    
            val_metrics_dict, cm = val_epoch(epoch, model, criterion, 
                    val_loader, num_classes, device)
            wandb.log({**train_metrics_dict, **val_metrics_dict})
            log_info = {
              'exp_name':exp_name,
              **train_metrics_dict, **val_metrics_dict
              }

            log_json_file = '/content/drive/MyDrive/Uni/CSE3001/Assignment/logs/logs.json'
            with open(log_json_file, 'a') as json_file:
              json.dump(log_info, json_file, indent=4)

    finally:
        run.finish()
        # pass


    # Plot confusion matrix from results of last val epoch
    # TODO Task 1c - call plot_confusion_matrix with appropriate arguments.
    plot_confusion_matrix(cm, class_names)
  