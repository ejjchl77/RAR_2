#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 11:06:39 2023

@author: ejjchl77
"""
import sklearn

met_model = '/home/ejjchl77/RAR_2/model/1114_trained_classifier.pickle'
img_path = '/home/ejjchl77/Desktop/EJ_PMV/Mockup_MET_heating/sub01_M/PMV_DCL_control/image/'
out_path = '/home/ejjchl77/Desktop/EJ_PMV/Mockup_MET_heating/sub01_M/PMV_DCL_control/met_result/'
runfile('s5_test.py', args= '--model_path %s --data_type folder --data_path %s --output_folder %s' %(met_model, img_path, out_path))

#%run run.py --model=mobilenet_thin --resize=432x368 --image=/images/p1.jpg
#runfile('run.py', args='--model=mobilenet_thin --resize=432x368 --image=./images/p1.jpg')

#%run s5_test.py  --model_path $met_model --data_type folder --data_path $img_path --output_folder $out_path