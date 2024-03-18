from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from config import Config
import os
from PIL import Image
import numpy as np

config = Config()
randomCrop = transforms.RandomCrop(config.input_size)
centerCrop = transforms.CenterCrop(config.input_size)
toTensor   = transforms.ToTensor()
class UnlabeledDataset(Dataset):
  """
  Returns a dataset of images, and the get method returns the normalized image as a [1,512,512] tensor,
  its transform in the same shape and the seed used for the transforms
  """

  def __init__(self,set,split,transform,image_dir,transform_crop = 0):
    self.image_dir = image_dir
    images = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith('.png')]
    images = sorted(images, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    unlabeled_idx = list(range(400,2000))
    self.mode = set
    self.transform_crop = transform_crop
    split_stop = int((split) * len(unlabeled_idx))
    if set == "val":
      self.image_files = images[:split_stop]
    if set == "train":
      self.image_files = images[split_stop:]
    self.transform = transform

  def __getitem__(self,idx):
    img = Image.open(self.image_files[idx]) # So far it's still grayscale
    output = img.copy()
    if self.mode == "train" and self.transform_crop > 0:
        output = randomCrop(img)
    #input = toTensor(centerCrop(input))
    input = toTensor(img)
    output = toTensor(output)
    return input,output

  def __len__(self):
    return len(self.image_files)



class DatasetLabeled(Dataset):
    def __init__(self, mode, val_ratio,train_image_path,label_image_path) :
        super().__init__()
        self.samples = []
        labelled_idx = list(range(350))
        idx_split = int((1-val_ratio)*350)
        images = [ file for file in os.listdir(train_image_path) if file.endswith('.png')]
        images = sorted(images, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        np.random.shuffle(labelled_idx)
        if mode == 'train' :
            labelled_idx = labelled_idx[:idx_split]
        elif mode == 'val' :
            labelled_idx = labelled_idx[idx_split:]
        else: # Test
           labelled_idx = [350,400]
        for i in labelled_idx :
            self.samples.append((os.path.join(train_image_path, f'{i}.png'),
                                os.path.join(label_image_path, f'{i}.png')))

    def __getitem__(self, index):
        path_img, path_label = self.samples[index]
        img = Image.open(path_img)
        label = Image.open(path_label)
        img = toTensor(img)
        label = toTensor(label)
        return img, label

    def __len__(self):
        return len(self.samples)
    

labeled_dataset = DatasetLabeled('train',0.2,'data/train/images','data/train/images')
len(labeled_dataset)
image,label = labeled_dataset[0]
print(image.shape,label.shape)