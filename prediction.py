
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import numpy as np
import torch
import cv2
import os
def prepare_plot(origImage, origMask, predMask):
  figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
  ax[0].grid(False)
  ax[1].grid(False)
  ax[2].grid(False)
  ax[0].imshow(origImage.permute(1, 2, 0) ) 
  ax[1].imshow(origMask )
  ax[2].imshow(predMask.permute(1, 2, 0))
  ax[0].set_title("Image")
  ax[1].set_title("Original Mask")
  ax[2].set_title("Predicted Mask")
  figure.tight_layout()
  figure.show()

def make_predictions(model, imagePath, masked_path,input_transform):
	# set model to evaluation mode
	model.eval()
	with torch.no_grad():		
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		# check to see if we are applying any transformations
		if input_transform is not None:
			# apply the transformations to both image and its mask
			image = input_transform(image) #[3,192, 256]

		#org = np.transpose(image, (1,2,0)) #[192, 256, 3]
		#image = Image.fromarray(image)
		#org = torch.from_numpy(image).long()

		#orig = image.copy()
		filename = imagePath.split(os.path.sep)[-1]
		filename=filename.replace("jpg","png")
		groundTruthPath = os.path.join(masked_path,filename)
		gtMask = cv2.imread(groundTruthPath)
		gtMask = cv2.resize(gtMask, (256, 192)) # (192, 256, 3)

	
		# image = np.transpose(image, (2, 0, 1))
		# image = np.expand_dims(image, 0)
		#image = torch.from_numpy(image)

		predMask = model(image.unsqueeze(0))   #.squeeze()
		
		#predMask = torch.softmax(predMask)
		predMask = predMask.cpu().numpy()#(1, 24, 192, 256)
		#print(predMask.shape)

	


		#print(np.unique(predMask))

		#print(predMask[predMask>0.5])

		#predMask = (predMask > 0.5) * 255
		#predMask = predMask.astype(np.uint8)
		#print(predMask[predMask>0.5])
		#print(np.unique(predMask))

		N,_, h, w = predMask.shape
		pred = predMask.transpose(0, 2, 3, 1).reshape(-1, 24).argmax(axis=1).reshape(N, h, w) #(1, 192, 256)
		pred=torch.from_numpy(pred)
		print(pred) 
		#pred = np.transpose(pred, (1,2,0)) # (192, 256,1)
		
		
		print(np.unique(pred))
		
		

		prepare_plot(image, gtMask, pred)
