# RecTorchCNN
 
Pentru a rula programul trebuie creat un mediu virtual.

Pentru  a crea mediu virtual se scrie urmatoare line in terminal in directorul proiectului: ```python -m venv venv```

Pentru a activa mediul virtual dupa ce sa creat se scrie urmatoarea linie: `. /Scritps/activate`

Sa se installeze urmatoarele librarii: import customtkinter
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from customtkinter import CTkImage
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
from PIL import Image, ImageTk
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torchviz
