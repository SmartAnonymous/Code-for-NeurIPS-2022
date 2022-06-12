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

if __name__ == '__main__':
	# Pred = predictor.predictor(device_name = "cuda:1")

	# stripes = []
	# for i in range(12):
	# 	stripes.append([])

	# for l in tqdm(range(100)):
	# 	label = Pred.cat_labels[10 * l]
	# 	class_dir = val_dir + label
	# 	imgs_path = sorted(os.listdir(class_dir))

	# 	for k in range(50):
	# 		img_path = class_dir + "/" + imgs_path[k]
	# 		img = Image.open(img_path).convert('RGB')
	# 		tensor = Pred.transform(img).unsqueeze(0).to(Pred.device)
	# 		prob, cat, attns, xs = Pred.predict_one("", tensor = tensor)#, zero_lines = zero_lines)
	# 		# print(prob, cat)
	# 		for i in range(12):
	# 			attn_i = np.mean(attns[i, 0], axis = 1)
	# 			stripes[i].append(attn_i)

	# stripes = np.array(stripes)
	# np.save("stripes.npy", stripes)

	# for i in range(12):
	# 	print(np.var(stripes[i]))

	stripes = np.load("stripes.npy")
	var_list = []
	for i in range(12):
		var_list.append(np.var(-np.log10(stripes[i])))
	var_list = np.array(var_list)
	print(var_list)

	fig = plt.figure()
	sns.set()
	sns.lineplot(x = range(12), y = var_list, color = "green")
	sns.scatterplot(x = range(12), y = var_list, color = "orange")
	plt.savefig("num_dist-log-var_list.png")

	# print(stripes.shape)
	# fig = plt.figure()
	# sns.set()
	# sns.set_style("white")
	# for i in range(12):
	# 	sns.distplot(-np.log10(stripes[i, ]), label = "Layer " + str(i),
	# 		color = sns.color_palette("RdYlBu", 12)[i])
	# plt.legend()
	# plt.savefig("num_dist/all" + str(i) +".png")

	# stripes = []
	
	# for l in range(100):
	# 	label = Pred.cat_labels[l]
	# 	stripe = np.load("var-0/" + label + ".npy")
	# 	# stripes.append(stripe)
	# 	# print(np.sum(stripe > 1e-2))
	# 	for i in range(50):
	# 		topk = np.argsort()[-(k + 1):]

	# stripes = np.array(stripes)
	# fig = plt.figure()
	# sns.set()
	# sns.distplot(-np.log10(stripes))
	# plt.savefig("num_dist.png")
