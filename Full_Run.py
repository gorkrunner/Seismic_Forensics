#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:49:18 2017

@author: panderso
"""

from Configure_Class import Configure_System
from GradientBoosting_Class import Classifier
from Data_Processing_Class import Data_Processing
from Data_Set_Class import Data_Set
import numpy as np
from sklearn.model_selection import train_test_split
import time
import timeit
from scipy.stats import norm as sp_norm
from scipy.stats import halfnorm as sp_halfnorm
from scipy.stats import randint as sp_randit
import os
import sys

def Timeit_Run():
    list_num = np.random.randint(0,3)
    row_num = np.random.randint(0,total_num)
    row = test_data_list[list_num][row_num,:]
    return classifier_dict['NODE_0'].Predicting(row.reshape(-1,1))  

def Node_DeBug(clf,test_data):
    
    node_results = []
    for row in test_data:
        node_results.append(clf.Predicting(row))
    
    return node_results

if __name__ == '__main__': 

    fname = raw_input('Enter config file name or location: ')
    timeit_run = raw_input('Do you want timeit to run: ')
    config = Configure_System(fname)
    config.Configure_Pull()
    
    data_process = Data_Processing(config.section_dict['DATA']['fft'],config.section_dict['DATA']['detrend'],config.section_dict['DATA']['norm'])
    seismic_data = os.path.join(os.getcwd(),config.section_dict['DATA']['seismic_data'])
    explosion_data = os.path.join(os.getcwd(),config.section_dict['DATA']['explosion_data'])
    nuclear_data = os.path.join(os.getcwd(),config.section_dict['DATA']['nuclear_data'])
    location_of_data = [seismic_data,explosion_data,nuclear_data]
    
    print 'Pulling in Data'
    data_set_class = Data_Set(location_of_data)
    
    classifier_dict = {}
    classifier_list = []
    data_dict = {}
    data_list = []
    
    
    for key in config.section_dict.keys():
        if 'DATA' in key:
            classifier_list.append(key)
        if 'NODE' in key:
            x_data = np.zeros((1,4096))
            y_data = np.zeros((1,))
            node_vals = eval(config.section_dict[key]['branch_vals'])
            
            data_set = data_set_class.Make_Data_Set(eval(config.section_dict[key]['percentage_list']))
            
            x_data = data_set[:,:4096]
            y_data = data_set[:,-1]
            
            x_processed = data_process.Process(x_data,1) 

            print 'Building {}'.format(key)
            if config.section_dict['DATA']['validate_size'] != str(0):
                print 'HyperParameter Optimization for {}'.format(key)
                x_train,x_validate,y_train,y_validate = train_test_split(x_processed,y_data,
                                                                 train_size = float(config.section_dict['DATA']['train_size']),
                                                                 test_size = float(config.section_dict['DATA']['validate_size']))
               
                node_name = Classifier(config.section_dict[key]['classifier'],eval(config.section_dict[key]['num_iter']))
                node_name.Build(config.section_dict[key]['hyper_search'],eval(config.section_dict[key]['param']),x_validate,y_validate)
                
                node_name.Fitting(x_train,y_train)
                
                
        
            if config.section_dict['DATA']['validate_size'] == str(0):
                node_name = Classifier(config.section_dict[key]['classifier'])
                node_name.Build(config.section_dict[key]['hyper_search'],eval(config.section_dict[key]['param']),None,None)
                node_name.Fitting(x_processed,y_data)
            
            classifier_dict.update({key:node_name})
            classifier_list.append(key)
            
            if key == 'NODE_0':
                data_list = [x_processed]
    sys.exit()            
    results_actual = []  

    print 'Setting Node Outputs'              
    for clf in reversed(sorted(classifier_list)):
        clf_dict = eval(config.section_dict[clf]['branch_dict'])
        for val,item in clf_dict.iteritems():
            if config.section_dict[clf]['terminal'] == str(False):
                if clf != 'DATA':
                    clf_dict[val][0] = classifier_dict[clf_dict[val][0]]
                elif clf == 'DATA':
                    data_process.Set_Node(eval(config.section_dict['DATA']['branch_dict']),config.section_dict['DATA']['terminal'])
            results_actual.append(item[1])                
            
        classifier_dict[clf].Set_Node(clf_dict,eval(config.section_dict[clf]['terminal']))
        
    print 'Making Virgin Data Sets'
    test_data_list = []
    y_data_list = []
    total_num = 0
    event_labels_dict = eval(config.section_dict['DATA']['event_labels'])
    for val in event_labels_dict.keys():
        if val == 0:
            x_virgin = data_set_class.Make_Virgin_Seismic() 
            x_virgin_processed = data_process.Process(x_virgin,1)
            y_virgin = 0
            seismic_num = x_virgin.shape[0]
        if val == 1:
            x_virgin = data_set_class.Make_Virgin_Explosion()
            x_virgin_processed = data_process.Process(x_virgin,1)
            y_virgin = 1
            exp_num = x_virgin.shape[0]
        if val == 2:
            x_virgin = data_set_class.Make_Virgin_Nuclear()
            x_virgin_processed = data_process.Process(x_virgin,1)
            y_virgin = 2
            nuc_num = x_virgin.shape[0]
        test_data_list.append(x_virgin_processed)
        y_data_list.append(y_virgin)
        total_num += len(x_virgin_processed)       
        
        
    #test_data_list = [x_seismic_virgin_processed,x_exp_virgin_processed,x_nuc_virgin_processed]
    #total_num = len(x_seismic_virgin_processed)+len(x_exp_virgin_processed)+len(x_nuc_virgin_processed)
    
#    virgin_data_set = np.vstack((x_seismic_virgin_processed,x_exp_virgin_processed))
#    virgin_data_set = np.vstack((virgin_data_set,x_nuc_virgin_processed))
#    num_rows = virgin_data_set.shape[0]
    
    print 'Testing Virgin Data Set'
    results_list = []
    incorrect_list = []
    total_tested = 0
    incorrect_label = np.zeros((3))
    time_array = np.zeros((total_num))
    for i,data_set in enumerate(test_data_list):
        for row in data_set:
            if config.section_dict['MAIN']['numnodes'] != str(1):  
                start_time = time.time()
                results_classifier = classifier_dict['NODE_0'].Predicting(row)
                end_time = time.time()

                if y_data_list[i] == 0:
                    if 'Seismic' not in results_classifier:
                        incorrect_label[0]+=1
                        continue                                       
                if y_data_list[i] == 1 and 'Explosion' in results_actual:
                    if 'Explosion' not in results_classifier:
                        incorrect_label[1]+=1
                        continue
                if y_data_list[i] == 1 and 'Chemical' in results_actual:
                    if 'Chemical' not in results_classifier:
                        incorrect_label[1]+=1
                        continue
                if y_data_list[i] == 2 and 'Explosion' in results_actual:
                    if 'Explosion' not in results_classifier:
                        incorrect_label[2]+=1
                        continue                    
                if y_data_list[i] == 2 and 'Nuclear' in results_actual:
                    if 'Nuclear' not in results_classifier:
                        incorrect_label[2]+=1
                        continue  
            else:
                start_time = time.time()
                results_classifier = classifier_dict[classifier_list[0]].Predicting(row)
                end_time = time.time()
                
                results_list.append(results_classifier)
                                     
                time_array[total_tested] = end_time - start_time                               
                total_tested+=1
                
                
                if y_data_list[i] == 0:
                    if 'Seismic' not in results_classifier:
                        incorrect_label[0]+=1
                        continue                                       
                if y_data_list[i] == 1 and 'Explosion' in results_actual:
                    if 'Explosion' not in results_classifier:
                        incorrect_label[1]+=1
                        continue
                if y_data_list[i] == 1 and 'Chemical' in results_actual:
                    if 'Chemical' not in results_classifier:
                        incorrect_label[1]+=1
                        continue
                if y_data_list[i] == 2 and 'Explosion' in results_actual:
                    if 'Explosion' not in results_classifier:
                        incorrect_label[2]+=1
                        continue                    
                if y_data_list[i] == 2 and 'Nuclear' in results_actual:
                    if 'Nuclear' not in results_classifier:
                        incorrect_label[2]+=1
                        continue

            
    
    correct_counter = total_tested - np.sum(incorrect_label)
    print 'Percentage Classifier Correctly'
    print '%.2f'%(correct_counter / float(total_tested)*100)
    
    print 'Incorrectly labeled by type'
    print 'Seismic {} out of {}'.format(str(incorrect_label[0]),seismic_num)
    print 'Chemical {} out of {}'.format(str(incorrect_label[1]),exp_num)
    print 'Nuclear {} out of {}'.format(str(incorrect_label[2]),nuc_num)
    
    print ' '
    print 'Average time to predict {}'.format(np.average(time_array))
    
    
    if timeit_run == 'yes':
        print ' '
        print 'Running Timeit'
        print timeit.timeit('Timeit_Run()',setup="from __main__ import Timeit_Run")
