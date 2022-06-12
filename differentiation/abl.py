import os
import sys
sys.path.append("..") 
from utils import predictor
import time
import numpy as np
from tqdm import tqdm

val_dir = "/home/workthu/zy/dataset/imagenet/imagenet_valset/"

if __name__ == '__main__':
	Pred = predictor.predictor(device_name = "cuda:1", model_name = "resnet101", log_dir = "log/resnet101")

	
	if not os.path.exists("each/"):
			os.makedirs("each/")

	for l in tqdm(range(11, 23)):
		# Resnet
		layers = [[], [], np.arange(l, 23).tolist(), np.arange(0, 3).tolist()]
		zero_lines_list = []
		for i in range(197):
			zero_lines = [[], [], np.arange(i).tolist(), np.arange(int(i / 4)).tolist(), ]
			zero_lines_list.append([layers, zero_lines])

		# RegNet
		# zero_lines_list = []
		# for i in range(197):
		# 	zero_lines_list.append([[], [], [np.arange(l, 11).tolist(), np.arange(i).tolist()], [[0], np.arange(int(i / 4)).tolist()]])

		accs = Pred.predict_val(val_dir, zero_lines_list = zero_lines_list, log_title = str(l) + "s3", num_classes = 200)
		np.save("each/resnet101-" + str(l) + ".npy", accs)

		# ViT
		# layers = np.arange(l, 12).tolist()
		# zero_heads = np.arange(0, 12).tolist()
		# zero_lines_list = []
		# for k in range(1, 578):
		# 	zero_lines_list.append([layers, zero_heads, np.arange(1, k).tolist()])

		# accs = Pred.predict_val(val_dir, zero_lines_list = zero_lines_list, log_title = str(l), num_classes = 200)
		# np.save("each/vit_base_patch16_384-" + str(l) + ".npy", accs)