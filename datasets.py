import collections
import csv
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

# TODO Task 1b - Implement LesionDataset
#        You must implement the __init__, __len__ and __getitem__ methods.
#
#        The __init__ function should have the following prototype
#          def __init__(self, img_dir, labels_fname):
#            - img_dir is the directory path with all the image files
#            - labels_fname is the csv file with image ids and their 
#              corresponding labels
#
#        Note: You should not open all the image files in your __init__.
#              Instead, just read in all the file names into a list and
#              open the required image file in the __getitem__ function.
#              This prevents the machine from running out of memory.
#
# TODO Task 1e - Add augment flag to LesionDataset, so the __init__ function
#                now look like this:
#                   def __init__(self, img_dir, labels_fname, augment=False):
#

class LesionDataset(torch.utils.data.Dataset):
  """
      img_dir is the directory path with all the image files
      labels_fname is the csv file with image ids and their 
  """
  def __init__(self, img_dir, labels_fname, augment=False, five_crop=False):
      self.img_dir = img_dir
      self.labels_fname = labels_fname
      self.df = pd.read_csv(labels_fname)
      self.images = self.df['image']
      self.labels = self.df.values[:, 1:].argmax(axis=1)
      self.classes = list(self.df.columns[1:])
      self.augment = augment
      self.five_crop = five_crop
      self.transform = transforms.Compose([
          # transforms.RandomHorizontalFlip(0.5),
          # transforms.RandomVerticalFlip(0.5),
          # transforms.RandomErasing(),
          transforms.RandomRotation(20),
          # transforms.RandomCrop((300,400)),
          transforms.Resize((450, 600))
        ])
      self.fivecrop = transforms.Compose(
        [
          transforms.FiveCrop((200,300)),
          transforms.Lambda(lambda crops: torch.stack([transforms.Resize((450, 600))(crop) for crop in crops]))
        ]
      )

  def __len__(self):
      return len(self.images)

  def __getitem__(self, idx):
      image = self.images[idx]
      label = self.labels[idx]

      image_path = self.img_dir + '/' + image + '.jpg'

      image = Image.open(image_path)
      image = torch.Tensor(np.array(image))
      image = image.permute(2,0,1)
      
      if self.augment:
          image = self.transform(image)

      if self.five_crop:
          image = self.fivecrop(image)

      return image, label

  def get_class_name(self, idx):
      return self.classes[idx]
    
  def get_all_class_names(self):
      return self.classes


    
