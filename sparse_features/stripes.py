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
	Pred = predictor.predictor(device_name = "cuda:1", model_name = "vit_base_patch16_224")

	layers_mean = np.zeros(12)
	for l in range(1):
		label = Pred.cat_labels[l]
		class_dir = val_dir + label
		imgs_path = sorted(os.listdir(class_dir))

		for k in range(1, 2):
			img_path = class_dir + "/" + imgs_path[k]
			img = Image.open(img_path).convert('RGB')
			tensor = Pred.transform(img).unsqueeze(0).to(Pred.device)
			prob, cat, attns, xs = Pred.predict_one("", tensor = tensor)#, zero_lines = zero_lines)
			print(prob, cat)
			# for i in range(12):
			# 	attn_i = np.mean(np.mean(attns[i, 0], axis = 0), axis = 0)
			# 	layers_mean[i] += np.sum(attn_i ** 2)

			attns = attns[:, 0, 0]
			fig = plt.figure(figsize = (54, 40))
			# plt.suptitle("Attention Maps", fontsize = 80)
			sns.set()
			xlabels = []
			for i in range(197):
				if i % 10 == 0 or i == 197:
					xlabels.append(str(i))
				else:
					xlabels.append("")
			for i in range(12):
				ax1 = fig.add_subplot(3, 4, i + 1)
				axesSub = sns.heatmap(-np.log(attns[i]), vmin = 0, vmax = 15, cbar = False, 
					xticklabels = False, yticklabels = False, square = True)
				# axesSub.set_title("Head " + chr(ord(str(i)) - ord("0") + ord("a")), fontsize = 30)
				axesSub.set_title("Layer " + str(i), fontsize = 60)
			fig.tight_layout()
			plt.savefig("attn_map-log" + str(k) + ".png")


		# fig2 = plt.figure()
		# sns.set()
		# sns.heatmap(np.zeros((0, 0)), vmin = 0, vmax = 15)
		# fig.tight_layout()
		# plt.savefig("colorbar" + str(k) + ".png")
			# xs = np.array(xs)
			# xs = xs - np.min(xs) + 1e-10
			# entro = entropy(xs, axis = 3).squeeze()
			# attn_mean = np.mean(np.mean(attns[:, 0], axis = 1), axis = 1)
			# print(entro.shape, attn_mean.shape)
			# fig = plt.figure(figsize = (10, 15))
			# sns.set()
			# ax1 = fig.add_subplot(2, 1, 1)
			# axesSub = sns.heatmap(-attn_mean)
			# axesSub.set_title("Attention Means", fontsize = 20)
			# ax1 = fig.add_subplot(2, 1, 2)
			# axesSub = sns.heatmap(entro)
			# axesSub.set_title("Output Entropy", fontsize = 20)
			# fig.tight_layout()
			# plt.savefig("attn_entropy.png")