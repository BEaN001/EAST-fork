import time
import torch
import subprocess
import os
from model import EAST
from detect import detect_dataset
import numpy as np
import shutil


def eval_model(model_name, test_img_path, submit_path, save_flag=True):
	if os.path.exists(submit_path):
		shutil.rmtree(submit_path) 
	os.mkdir(submit_path)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = EAST(False).to(device)
	model.load_state_dict(torch.load(model_name))
	model.eval()
	
	start_time = time.time()
	detect_dataset(model, device, test_img_path, submit_path)
	os.chdir(submit_path)
	res = subprocess.getoutput('zip -q submit.zip *.txt')
	res = subprocess.getoutput('mv submit.zip ../')
	os.chdir('../')
	res = subprocess.getoutput('python ./evaluate/script.py –g=./evaluate/gt.zip –s=./submit.zip')
	print(res)
	os.remove('./submit.zip')
	print('eval time is {}'.format(time.time()-start_time))	

	if not save_flag:
		shutil.rmtree(submit_path)


if __name__ == '__main__': 
	model_name = '/home/yubin/pro/EAST/pths/east_vgg16.pth'
	# model_name = './pths/model_epoch_600.pth'
	test_img_path = os.path.abspath('/data/yubindata/ICDAR2015/ch4_test_images')
	submit_path = './submit'
	eval_model(model_name, test_img_path, submit_path)
	# epoch300: Calculated!{"precision": 0.7948841698841699, "recall": 0.7929706307173808, "hmean": 0.7939262472885033, "AP": 0}
	# epoch400: Calculated!{"precision": 0.8115384615384615, "recall": 0.8127106403466539, "hmean": 0.8121241279769065, "AP": 0}
	# epoch500: Calculated!{"precision": 0.8277043563387175, "recall": 0.8141550312951372, "hmean": 0.8208737864077669, "AP": 0}
	# epoch600: Calculated!{"precision": 0.8103696591454633, "recall": 0.8127106403466539, "hmean": 0.8115384615384617, "AP": 0}
	# east_vgg16.pth: Calculated!{"precision": 0.8435782108945528, "recall": 0.8127106403466539, "hmean": 0.8278567925453654, "AP": 0}

