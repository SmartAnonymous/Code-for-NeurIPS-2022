import os
import sys
sys.path.append("..") 
from utils import predictor
import time
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
	results = []
	errs = []
	# for i in range(3):
	# 	accs = np.load("each/resnet50" + str(i) + ".npy")
	# 	result = np.sum(accs - accs[-1]) / accs.shape[0] / (accs[0] - accs[-1])
	# 	err = accs[-1] / accs[0]
	# 	print(i, result, err)
	# 	results.append(result)
	# 	errs.append(err)

	# for i in range(6):
	# 	accs = np.load("each/resnet50" + str(i) + "-s3.npy")
	# 	result = np.sum(accs - accs[-1]) / accs.shape[0] / (accs[0] - accs[-1])
	# 	err = accs[-1] / accs[0]
	# 	print(i, round(result, 3), round(err, 3))
	# 	results.append(result)
	# 	errs.append(err)

	for i in range(6, 12):
		accs = np.load("each/vit_base_patch32_224-" + str(i) + ".npy")
		result = np.sum(accs - accs[-1]) / accs.shape[0] / (accs[0] - accs[-1])
		err = accs[-1] / accs[0]
		# print(i, round(result, 3), round(err, 3))

		results.append(result)
		errs.append(err)

	a = np.array([results, errs]).T
	print(a)
	np.save("err-area/vit_base_patch32_224.npy", a)