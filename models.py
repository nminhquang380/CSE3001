import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

# TODO Task 1c - Implement a SimpleBNConv
class SimpleBNConv(nn.Module):
  def __init__(self,input_size=3*450*600, output_size=7):
    super().__init__()

    self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 2
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 3
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 4
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 5
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.seq = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
    )

  def forward(self, x):
    feature = self.features(x)
    return self.seq(feature)

class FiveCropConv(nn.Module):
  def __init__(self,input_size=3*450*600, output_size=7):
    super().__init__()

    self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 2
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 3
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 4
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 5
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.seq = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
    )

  def forward(self, x):
    feature = self.features(x)
    return self.seq(feature)


# TODO Task 1f - Create a model from a pre-trained model from the torchvision
#  model zoo.
def construct_resnet18(num_classes=7):
    # TODO: Download the pre-trained model
    # SOLUTION LINE
    resnet18 = models.resnet18(weights="IMAGENET1K_V1")

    # TODO: Freeze model weights
    # SOLUTION LINE
    for param in resnet18.parameters():
        param.requires_grad = False
    # TODO: Replace the final linear layer
    # SOLUTION LINE
    resnet18.fc = nn.Linear(512, num_classes)

    return resnet18

def construct_resnet50_v2(num_classes=7):
    # TODO: Download the pre-trained model
    # SOLUTION LINE
    resnet50 = models.resnet50(weights="IMAGENET1K_V2")

    # TODO: Freeze model weights
    # SOLUTION LINE
    for param in resnet50.parameters():
        param.requires_grad = False
    # TODO: Replace the final linear layer
    # SOLUTION LINE
    resnet50.fc = nn.Linear(2048, num_classes)

    return resnet50

def construct_efficientnet_v2_s(num_classes=7):
    efficientnet_s = models.efficientnet_v2_s(weights='IMAGENET1K_V1')

    for param in efficientnet_s.parameters():
      param.requires_grad = False

    efficientnet_s.classifier = nn.Sequential(
      nn.Dropout(0.2),
      nn.Linear(1280, 128),
      nn.ReLU(),
     
      nn.Linear(128, num_classes),
    )

    return efficientnet_s

def construct_densenet_121(num_classes=7):
    densenet = models.densenet121(weights='IMAGENET1K_V1')

    for param in densenet.parameters():
      param.requires_grad = False
    
    densenet.classifier = nn.Linear(1024, num_classes)

    return densenet


# TODO Task 1f - Create your own models
class DropoutConvNet(nn.Module):
  def __init__(self, input_size=450*600, output_size=7):
    super().__init__()

    self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 8, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Layer 2
            nn.Conv2d(8, 16, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Layer 3
            nn.Conv2d(16, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Layer 4
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Layer 5
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
    )
    self.seq = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(24576, 64),
            nn.ReLU(),

            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
    )

  def forward(self, x):
    feature = self.features(x)
    return self.seq(feature)

class DropoutBatchNormNet(nn.Module):
  def __init__(self, input_size=450*600, output_size=7):
    super().__init__()

    self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 8, kernel_size=3, padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Layer 2
            nn.Conv2d(8, 16, kernel_size=3, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Layer 3
            nn.Conv2d(16, 32, kernel_size=3, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Layer 4
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Layer 5
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
    )
    self.seq = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(24576, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
    )

  def forward(self, x):
    feature = self.features(x)
    return self.seq(feature)


class DeepConvNet(nn.Module):
  def __init__(self, input_size=450*600, output_size=7):
    super().__init__()

    self.features = nn.Sequential(
        # Layer 1
        nn.Conv2d(3, 8, kernel_size=3, padding=1, stride=2),
        # nn.BatchNorm2d(8),
        nn.ReLU(),

        # Layer 2
        nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=2),
        # nn.BatchNorm2d(16),
        nn.ReLU(),

        # Layer 3
        nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
        # nn.BatchNorm2d(32),
        nn.ReLU(),

        # Layer 4
        nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
        # nn.BatchNorm2d(64),
        nn.ReLU(),
        )

    self.seq = nn.Sequential(
            nn.Flatten(),
            nn.Linear(70528, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
    )

  def forward(self, x):
    feature = self.features(x)
    return self.seq(feature)

class DeepConvBatchNormNet(nn.Module):
  def __init__(self, input_size=450*600, output_size=7):
    super().__init__()

    self.features = nn.Sequential(
        # Layer 1
        nn.Conv2d(3, 8, kernel_size=3, padding=1, stride=2),
        nn.BatchNorm2d(8),
        nn.ReLU(),

        # Layer 2
        nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=2),
        nn.BatchNorm2d(16),
        nn.ReLU(),

        # Layer 3
        nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
        nn.BatchNorm2d(32),
        nn.ReLU(),

        # Layer 4
        nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        )

    self.seq = nn.Sequential(
            nn.Flatten(),
            nn.Linear(70528, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
    )

  def forward(self, x):
    feature = self.features(x)
    return self.seq(feature)

class SkipBlock(nn.Module):
  def __init__(self, input_channels, output_channels):
    super().__init__()
    self.layer1 = nn.Sequential(
      nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
      # nn.BatchNorm2d(output_channels),
      nn.ReLU(),
    )
    self.layer2 = nn.Sequential(
      nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
      # nn.BatchNorm2d(output_channels),
      nn.ReLU(),
    )
  
  def forward(self, x):
    x2 = self.layer1(x)
    x3 = self.layer2(x2)
    return x2+x3

class SkipBNBlock(nn.Module):
  def __init__(self, input_channels, output_channels):
    super().__init__()
    self.layer1 = nn.Sequential(
      nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(output_channels),
      nn.ReLU(),
    )
    self.layer2 = nn.Sequential(
      nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(output_channels),
      nn.ReLU(),
    )

  def forward(self, x):
    x2 = self.layer1(x)
    x3 = self.layer2(x2)
    return x2+x3
    
    
class SkipNet(nn.Module):
  def __init__(self, input_size=450*600, output_size=7):
    super().__init__()
    
    self.features = nn.Sequential(
      SkipBlock(3, 8),
      nn.MaxPool2d(2),
      SkipBlock(8, 16),
      nn.MaxPool2d(2),
      SkipBlock(16, 32),
      nn.MaxPool2d(2),
      # SkipBlock(32, 64),
    )

    self.seq = nn.Sequential(
      nn.Flatten(),
      nn.Linear(134400, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, output_size)
    )

  def forward(self, x):
    feature = self.features(x)
    return self.seq(feature)
    
class SkipBNNet(nn.Module):
  def __init__(self, input_size=450*600, output_size=7):
    super().__init__()
    
    self.features = nn.Sequential(
      SkipBNBlock(3, 8),
      nn.MaxPool2d(2),
      SkipBNBlock(8, 16),
      nn.MaxPool2d(2),
      SkipBNBlock(16, 32),
      nn.MaxPool2d(2),
      # SkipBNBlock(32, 64),
    )

    self.seq = nn.Sequential(
      nn.Flatten(),
      nn.Linear(134400, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, output_size)
    )

  def forward(self, x):
    feature = self.features(x)
    return self.seq(feature)

# class MaxPoolSkipBNNet(nn.Module):
#   def __init__(self, input_size=450*600, output_size=7):
#     super().__init__()
    
#     self.features = nn.Sequential(
#       SkipBNBlock(3, 8),
#       nn.MaxPool2d(2),
#       SkipBNBlock(8, 16),
#       nn.MaxPool2d(2),
#       SkipBNBlock(16, 32),
#       nn.MaxPool2d(2),
#       SkipBNBlock(32, 64),
#       nn.MaxPool2d(2),
#     )

#     self.seq = nn.Sequential(
#       nn.Flatten(),
#       nn.Linear(32256, 64),
#       nn.ReLU(),
#       nn.Linear(64, 32),
#       nn.ReLU(),
#       nn.Linear(32, output_size)
#     )

#   def forward(self, x):
#     feature = self.features(x)
#     return self.seq(feature)

class DropOutSkipBNNet(nn.Module):
  def __init__(self, input_size=450*600, output_size=7):
    super().__init__()
    
    self.features = nn.Sequential(
      SkipBNBlock(3, 8),
      nn.MaxPool2d(2),
      SkipBNBlock(8, 16),
      nn.MaxPool2d(2),
      SkipBNBlock(16, 32),
      nn.MaxPool2d(2),
      # SkipBNBlock(32, 64),
    )

    self.seq = nn.Sequential(
      nn.Flatten(),
      nn.Dropout(0.4),
      nn.Linear(134400, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, output_size)
    )

  def forward(self, x):
    feature = self.features(x)
    return self.seq(feature)

      

