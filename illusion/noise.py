import os
import sys
sys.path.append("..") 
from utils import predictor
import time
import numpy as np
from tqdm import tqdm
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

val_dir = "/home/workthu/zy/dataset/imagenet/imagenet_valset/"
np.random.seed(2021)

def gauss_noise(img, var = 0):
	img = np.array(np.array(img) / 255, dtype = float)
	img = img + np.random.normal(0, var ** 0.5, img.shape)
	img = np.uint8(np.clip(img, 0.0, 1.0) * 255)
	return Image.fromarray(img)


if __name__ == '__main__':
	Pred = predictor.predictor(device_name = "cuda:0", model_name = "vit_base_patch16_224")
	var_list = [0]#[0.2, 0.3, 0.4, 0.75]

	for var in var_list:
		print("Gauss Noise: Var=", var)
		if not os.path.exists("preds-vit-base/var-" + str(var)):
			os.makedirs("preds-vit_base_patch32_224/var-" + str(var))

		for l in tqdm(range(1000)):
			label = Pred.cat_labels[l]
			class_dir = val_dir + label
			imgs_path = sorted(os.listdir(class_dir))

			pred = []
			stripe = []
			for k in range(50):
				img_path = class_dir + "/" + imgs_path[k]
				img = Image.open(img_path).convert('RGB')
				img1 = gauss_noise(img, var = var)
				tensor = Pred.transform(img1).unsqueeze(0).to(Pred.device)
				# prob, cat, attns, xs = Pred.predict_one("", tensor = tensor)
				prob, cat = Pred.predict_one("", tensor = tensor)
				pred.append(cat)
				# attn = np.mean(attns[11, 0], axis = 1)
				# stripe.append(attn)

			pred = np.array(pred)
			# stripe = np.array(stripe)
			np.save("preds-vit_base_patch32_224/var-" + str(var) + "/" + label + ".npy", pred)
			# np.save("stripes-gauss-large/var-" + str(var) + "/" + label + ".npy", stripe)