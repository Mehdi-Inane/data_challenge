
from dataset import *
from utils import *
import torch
from model import *
from loss_labeled import *

from meanshift import *

img_path = 'data/train/images'
train_dataset = DatasetLabeled('train',0.2,'data/train/images','data/data/train/y_train')
val_dataset = DatasetLabeled('val',0.2,'data/train/images','data/data/train/y_train')
test_dataset = DatasetLabeled('test',0.2,'data/train/images','data/data/train/y_train')
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size,num_workers = 4, shuffle=True)
val_dataloader   = torch.utils.data.DataLoader(val_dataset,   batch_size=4,num_workers = 4, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset,   batch_size=4,num_workers = 4, shuffle=False)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
autoencoder = WNet().to('cuda')
model_base_name = 'model_checkpoints/WNET_supervised_'
if config.resume:
        autoencoder = torch.load(config.ckpt).to('cuda')
###Training
optimizerE = torch.optim.Adam(autoencoder.U_encoder.parameters(), lr=0.003)
meanshift = MeanShiftCluster()
autoencoder.train()
criterion = GlobalLoss().to('cuda')
for epoch in range(config.num_epochs):
    autoencoder.train()
    running_loss = 0
    for image,ground_truth in train_dataloader:
        if torch.cuda.is_available():
            image  = image.cuda()
            ground_truth = ground_truth.cuda()
        optimizerE.zero_grad()
        segmentations = autoencoder.forward_encoder(image)
        print(segmentations.shape)
        segmentations = gumbel_softmax(segmentations)
        loss = criterion(segmentations,ground_truth)
        loss.backward()
        optimizerE.step()
        running_loss += loss
    running_loss = running_loss / len(train_dataloader)
    print('Training loss: ',running_loss)
    # Computing validation loss
    with torch.no_grad:
        val_run_loss = 0
        for image, ground_truth in val_dataloader:
            if torch.cuda.is_available():
                image  = image.cuda()
                ground_truth = ground_truth.cuda()
            segmentations = autoencoder.forward_encoder(image)
            segmentations = gumbel_softmax(segmentations)
            loss = criterion(segmentations,ground_truth)
            val_run_loss += loss
        val_run_loss = val_run_loss / len(val_dataloader)
        print('Validation Error: ',val_run_loss)
    # Computing Rand Index on test set
    if epoch % 5 == 0:
        torch.save(autoencoder,f'{model_base_name + str(epoch + config.resume_epoch + 1)}.pth')
        # Computing rand index
        rand_index = compute_metric(autoencoder) 
        print(f'Rand index in epoch {epoch}: {rand_index}')
              
              
            



