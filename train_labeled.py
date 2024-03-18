
from dataset import *
from utils import *
import torch
from model import *
from loss_labeled import *
from loss import *
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
autoencoder = torch.nn.DataParallel(autoencoder)
model_base_name = 'model_checkpoints/WNET_supervised_'
if config.resume:
        autoencoder = torch.load(config.ckpt).to('cuda')
        autoencoder = torch.nn.DataParallel(autoencoder)
###Training
optimizerE = torch.optim.Adam(autoencoder.module.U_encoder.parameters(), lr=0.003)
optimizerW = torch.optim.Adam(autoencoder.module.parameters(), lr=0.003)
meanshift = MeanShiftCluster()
autoencoder.train()
criterion = GlobalLoss().to('cuda')
ncutloss_layer = NCutLoss2D()
progress_images, progress_expected = next(iter(val_dataloader))
for epoch in range(config.num_epochs):
    autoencoder.train()
    running_loss = 0
    i = 0
    for image,ground_truth in train_dataloader:
        if torch.cuda.is_available():
            image  = image.cuda()
            ground_truth = ground_truth.cuda()
        optimizerE.zero_grad()
        segmentations = autoencoder.module.forward_encoder(image)
        l_soft_n_cut  = ncutloss_layer(segmentations, image)
        l_soft_n_cut.backward(retain_graph=False)
        optimizerE.step()
        optimizerW.zero_grad()
        segmentations, reconstructions = autoencoder.forward(image)
        loss = criterion(reconstructions,ground_truth)
        loss.backward()
        optimizerW.step()
        if not loss :
            loss = 0
        if not l_soft_n_cut:
            l_soft_n_cut = 0
        if loss.shape != l_soft_n_cut.shape:
            loss = loss.squeeze(0)
        running_loss += (loss + l_soft_n_cut)
        if config.showSegmentationProgress and i == 0: # If first batch in epoch
            save_progress_image(autoencoder, progress_images, epoch)
        i +=1
    running_loss = running_loss / len(train_dataloader)
    print('Training loss: ',running_loss)
    # Computing validation loss
    with torch.no_grad():
        val_run_loss = 0
        for image, ground_truth in val_dataloader:
            if torch.cuda.is_available():
                image  = image.cuda()
                ground_truth = ground_truth.cuda()
            segmentations = autoencoder.module.forward_encoder(image)
            l_soft_n_cut  = ncutloss_layer(segmentations, image)
            segmentations, reconstructions = autoencoder.forward(image)
            loss = criterion(reconstructions,ground_truth)
            if loss.shape != l_soft_n_cut.shape:
                loss = loss.squeeze(0)
            val_run_loss += (loss + l_soft_n_cut)
        val_run_loss = val_run_loss / len(val_dataloader)
        print('Validation Error: ',val_run_loss)
    # Computing Rand Index on test set
    if epoch % 5 == 0:
        torch.save(autoencoder,f'{model_base_name + str(epoch + config.resume_epoch + 1)}.pth')
        # Computing rand index
        rand_index = compute_metric(autoencoder,meanshift,test_dataloader,device) 
        print(f'Rand index in epoch {epoch}: {rand_index}')
              
              
            



