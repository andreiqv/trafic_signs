#export CUDA_VISIBLE_DEVICES=3
import os

if os.path.exists('.local'):
	
	data_dir = '/mnt/lin2/datasets/splited/'
	num_epochs = 5
	batch_size = 4
	num_workers = 2

	SHOW_BAR = False
	DEBUG = False
	TOPk = 3

else:
	#data_dir = '/home/andrei/Data/Datasets/Scales/splited/'
	
	data_dir = '/home/andrei/work/t7_cv/trafic_signs/splited/'
	num_epochs = 25
	batch_size = 32
	num_workers = 8	

	SHOW_BAR = True
	DEBUG = False
	TOPk = 3
