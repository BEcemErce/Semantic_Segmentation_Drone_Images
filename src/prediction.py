
import matplotlib.pyplot as plt
from torchvision import transforms
import tensorflow as tf
import numpy as np
import torch
import cv2
import os
def make_predictions(model, image_path,masked_path,cmap):
    #preprocess the image

	image = cv2.imread(image_path, cv2.IMREAD_COLOR)
	image=image/255.0
	image=cv2.resize(image,(1024,512))
	image = image.astype(np.float32)
	tensor_transform=transforms.ToTensor()
	image=tensor_transform(image) #3, 800, 1200

	model.eval()
	with torch.no_grad():		
		pred = model(image.unsqueeze(0))                 # input:1,3,800,1200
		pred=torch.argmax(pred,dim=1)  #1, 800,1200

	#rgb mask
	filename = image_path.split(os.path.sep)[-1]
	filename=filename.replace("jpg","png")
	rgb_orig_mask = os.path.join(masked_path,filename)
	rgb_orig_mask = cv2.imread(rgb_orig_mask)[:, :, 0]
	rgb_orig_mask=cv2.resize(rgb_orig_mask,(1024, 512))
	print(rgb_orig_mask.shape)
	
	#i = 18
	pred=pred.squeeze(0).numpy() # 800,1200

	figure, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
	ax[0].grid(False)
	ax[1].grid(False)
	ax[0].imshow(cmap[pred])
	ax[1].imshow(cmap[rgb_orig_mask])
	ax[0].set_title('Prediction')
	ax[1].set_title('Ground truth')

	plt.show()
	