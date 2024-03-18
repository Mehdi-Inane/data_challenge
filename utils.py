import os
import torch
from config import Config
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import sklearn.metrics
config = Config()

def compute_metric(model,meanshift,data_loader,device):
  scores = []
  model.eval()  # Set the model to evaluation mode
  with torch.no_grad():
    for batch in (data_loader):
      for (real_image,labeled_image) in zip(*batch):
        segmentations,predict_logits = model.forward(real_image.to(device))
        clustered_preds = meanshift(predict_logits).flatten().cpu()
        labels_flat = labeled_image.flatten().cpu()
        score = sklearn.metrics.adjusted_rand_score(clustered_preds,labels_flat)
        scores.append(score)
  return np.mean(scores)


def clear_progress_dir(): # Or make the dir if it does not exist
    if not os.path.isdir(config.segmentationProgressDir):
        os.mkdir(config.segmentationProgressDir)
    else: # Clear the directory
        for filename in os.listdir(config.segmentationProgressDir):
            filepath = os.path.join(config.segmentationProgressDir, filename)
            os.remove(filepath)

def enumerate_params(models):
	num_params = 0
	for model in models:
		for param in model.parameters():
			if param.requires_grad:
				num_params += param.numel()
	print(f"Total trainable model parameters: {num_params}")

def save_model(autoencoder, modelName):
    path = os.path.join("./models/", modelName.replace(":", " ").replace(".", " ").replace(" ", "_"))
    torch.save(autoencoder, path)
    with open(path+".config", "a+") as f:
        f.write(str(config))
        f.close()

def save_progress_image(autoencoder, progress_images, epoch):
    if not torch.cuda.is_available():
        segmentations, reconstructions = autoencoder(progress_images)
    else:
        segmentations, reconstructions = autoencoder(progress_images.cuda())

    f, axes = plt.subplots(4, config.val_batch_size, figsize=(8,8))
    for i in range(config.val_batch_size):
        segmentation = segmentations[i]
        pixels = torch.argmax(segmentation, axis=0).float() / config.k # to [0,1]

        axes[0].imshow(progress_images[i].permute(1, 2, 0))
        axes[1].imshow(pixels.detach().cpu())
        axes[2].imshow(reconstructions[i].detach().cpu().permute(1, 2, 0))
        if config.variationalTranslation:
            axes[3].imshow(progress_expected[i].detach().cpu().permute(1, 2, 0))
    plt.savefig(os.path.join(config.segmentationProgressDir, str(epoch)+".png"))
    plt.close(f)
