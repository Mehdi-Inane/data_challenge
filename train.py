from config import Config
from dataset import UnlabeledDataset
import torch
from loss import *
from model import WNet
import matplotlib.pyplot as plt
import pickle
from utils import *

config = Config()


#### Building datasets and dataloaders

img_path = 'data/train/images'
train_dataset = UnlabeledDataset('train',0.2,None,img_path,1)
val_dataset = UnlabeledDataset('val',0.2,None,img_path,1)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size,num_workers = 4, shuffle=True)
val_dataloader   = torch.utils.data.DataLoader(val_dataset,   batch_size=4,num_workers = 4, shuffle=False)


#### Model 
autoencoder = WNet().to('cuda')
ncutloss_layer = NCutLoss2D()
optimizerE = torch.optim.Adam(autoencoder.U_encoder.parameters(), lr=0.003)
optimizerW = torch.optim.Adam(autoencoder.parameters(), lr=0.003)
model_base_name = 'model_checkpoints/WNET_'
if config.resume:
        autoencoder = torch.load(config.ckpt).to('cuda')
###Training

autoencoder.train()
progress_images, progress_expected = next(iter(val_dataloader))
for epoch in range(config.num_epochs):
        running_loss = 0.0
        ncutloss = []
        reconloss = []
        for i, [inputs, outputs] in enumerate(train_dataloader, 0):

            if config.showdata:
                print(inputs.shape)
                print(outputs.shape)
                print(inputs[0])
                plt.imshow(inputs[0].permute(1, 2, 0))
                plt.show()

            if torch.cuda.is_available():
                inputs  = inputs.cuda()
                outputs = outputs.cuda()

            optimizerE.zero_grad()
            segmentations = autoencoder.forward_encoder(inputs)
            l_soft_n_cut  = ncutloss_layer(segmentations, inputs)
            l_soft_n_cut.backward(retain_graph=False)
            optimizerE.step()
            ncutloss.append(l_soft_n_cut)

            optimizerW.zero_grad()

            segmentations, reconstructions = autoencoder.forward(inputs)

            l_reconstruction = reconstruction_loss(
                inputs if config.variationalTranslation == 0 else outputs,
                reconstructions
            )
            reconloss.append(l_reconstruction)
            l_reconstruction.backward(
                retain_graph=False)  # We only need to do retain graph =true if we're backpropping from multiple heads
            optimizerW.step()
            if config.debug and (i%50) == 0:
                print(i)
            running_loss += l_reconstruction + l_soft_n_cut#loss.item()
            if config.showSegmentationProgress and i == 0: # If first batch in epoch
                save_progress_image(autoencoder, progress_images, epoch)

        epoch_loss = running_loss / len(train_dataloader.dataset)
        print(f"Epoch {epoch} loss: {epoch_loss:.6f}")

        if config.saveModel and (epoch % 5 == 0):
            torch.save(autoencoder,f'{model_base_name + str(epoch + config.resume_epoch + 1)}.pth')

        with open('n_cut_loss.pkl','ab') as f:
          pickle.dump(ncutloss, f)

        with open('reconstruction_loss.pkl','ab') as fp:
          pickle.dump(reconloss, fp)

