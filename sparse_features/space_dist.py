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
from scipy.stats import entropy

val_dir = "/home/workthu/zy/dataset/imagenet/imagenet_valset/"
np.random.seed(2021)

if __name__ == '__main__':
	# Pred = predictor.predictor(device_name = "cuda:1", model_name = "vit_base_patch16_224")

	# layers_mean = np.zeros((12, 197))
	# for l in tqdm(range(100)):
	# 	label = Pred.cat_labels[l]
	# 	class_dir = val_dir + label
	# 	imgs_path = sorted(os.listdir(class_dir))

	# 	for k in range(50):
	# 		img_path = class_dir + "/" + imgs_path[k]
	# 		img = Image.open(img_path).convert('RGB')
	# 		tensor = Pred.transform(img).unsqueeze(0).to(Pred.device)
	# 		prob, cat, attns, xs = Pred.predict_one("", tensor = tensor)#, zero_lines = zero_lines)
	# 		# print(prob, cat)
	# 		for i in range(12):
	# 			attn_i = np.mean(np.mean(attns[i, 0], axis = 0), axis = 0)
	# 			layers_mean[i] += attn_i

	# layers_mean /= 100 * 50
	# dist = layers_mean[11, 1:].reshape(14, 14)
	# np.save("spa_dist.npy", dist)

	dist = np.load("spa_dist.npy")
	# xlabels = []
	# for i in range(14):
	# 	xlabels.append(str(14 * i + 1) + " ~ " + str(14 * i + 14))

	# fig = plt.figure()
	# sns.set()
	# sns.heatmap(dist, square = True, xticklabels = False, yticklabels = xlabels)
	# plt.savefig("spa_dist.png")
	print(np.argsort(dist.reshape((196)))[-5:] + 1)

	# fig = plt.figure(figsize = (26, 16))
	# sns.set()
	# for i in range(12):
	# 	ax1 = fig.add_subplot(3, 4, i + 1)
	# 	axesSub = sns.heatmap(layers_mean[i, 1:].reshape(14, 14))
	# 	axesSub.set_title("Layer " + str(i), fontsize = 24)
	# fig.tight_layout()
	# plt.savefig("layers_mean.png")