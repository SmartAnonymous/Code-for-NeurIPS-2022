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

# val_dir = "/home/workthu/zy/dataset/imagenet/imagenet_valset/"

if __name__ == '__main__':
	Pred = predictor.predictor(device_name = "cpu") #, log_dir = "lines")
	mod = "vit_base_patch32_224"

	var_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1]

	paras = []

	if not os.path.exists("paras/"):
		os.makedirs("paras/")

	for i in range(len(var_list)):
		var = var_list[i]
		correct = 0
		stripes = []
		labels = np.zeros(1000)
		for l in range(1000):
			label = Pred.cat_labels[l]
			preds_path = "preds-" + mod + "/var-" + str(var) + "/" + label + ".npy"
			preds = np.load(preds_path)
			# correct += np.sum(preds == l)
			for i in range(50):
				if l != preds[i]:
					labels[preds[i]] += 1

		paras.append(np.sum(labels ** 2) / np.sum(labels) ** 2)
	
	paras = np.array(paras)
	print(paras)
	np.save("paras/" + mod + ".npy", paras)
		# print(var, np.sum(s ** 2) / np.sum(s) ** 2)

		# plt.figure()
		# sns.set()
		# sns.countplot(np.mean(labels, 0))
		# # sns.heatmap(labels, vmin = 0, vmax = 15)
		# # plt.xlabel("Predicted Labels")
		# # plt.ylabel("True Labels")
		# # plt.title("Confusion Matrix-" + mod + "-var" + str(var))
		# plt.savefig("confmat2/" + mod + "/var-" + str(var) + ".png")
		# plt.close()
		# print(var)