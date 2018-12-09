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

class Classifier_Run(object):
    
    def __init__(self,fname,timeit_run):
        self.config_fname = fname
        self.timit_run = timeit_run

#    def Timeit_Run(self):
#        list_num = np.random.randint(0,3)
#        row_num = np.random.randint(0,total_num)
#        row = test_data_list[list_num][row_num,:]
#        return classifier_dict['NODE_0'].Predicting(row.reshape(-1,1))  
    
    def Node_DeBug(self,clf,test_data):
        
        node_results = []
        for row in test_data:
            node_results.append(clf.Predicting(row))
        
        return node_results

    
    def Build_Classifiers(self):
    
        
        config = Configure_System(self.config_fname)
        config.Configure_Pull()
        
        self.config = config
        
        self.data_process = Data_Processing(config.section_dict['DATA']['fft'],
                                            config.section_dict['DATA']['detrend'],
                                           config.section_dict['DATA']['norm'],
                                              config.section_dict['DATA']['optimization'])
	
	seismic_loc = config.section_dict['DATA']['seismic_data']
	explosion_loc = config.section_dict['DATA']['explosion_data']
	nuclear_loc = config.section_dict['DATA']['nuclear_data']
        
        location_of_data = [seismic_loc,explosion_loc,nuclear_loc]
        
        print 'Pulling in Data'
        self.data_set_class = Data_Set(location_of_data)
        classifier_dict = {}
        classifier_list = []
        
        print 'Building Classifiers'
        for key in config.section_dict.keys():
            if 'DATA' in key:
                classifier_list.append(key)
                classifier_dict.update({key:self.data_process})
            if 'NODE' in key:

                #node_vals = eval(config.section_dict[key]['branch_vals'])
                
                print 'Building Data Set'
                data_set = self.data_set_class.Make_Data_Set(eval(config.section_dict[key]['percentage_list']))

                x_data = np.zeros((1,data_set.shape[1]-1))
                y_data = np.zeros((1,))
                
                x_data = data_set[:,:(data_set.shape[1]-1)]
                y_data = data_set[:,-1]
                
                print 'Processing Data Set'
                x_processed = self.data_process.Process(x_data,1) 
                
                print 'Building {}'.format(key)
                if config.section_dict[key]['hyper_search'] != 'none' and config.section_dict[key]['hyper_search'] != 'manual':
                    print 'HyperParameter Optimization for {}'.format(key)
                    x_train,x_validate,y_train,y_validate = train_test_split(x_processed,y_data,
                                                                     train_size = float(config.section_dict['DATA']['train_size']),
                                                                     test_size = float(config.section_dict['DATA']['validate_size']))
                   
                    node_name = Classifier(clf_type=config.section_dict[key]['classifier'],clf_opt=eval(config.section_dict[key]['optimization']),num_iter=eval(config.section_dict[key]['num_iter']))
                    #node_name = Classifier(config.section_dict[key]['classifier'],eval(config.section_dict[key]['num_iter']))
                    node_name.Build(config.section_dict[key]['hyper_search'],eval(config.section_dict[key]['param']),x_validate,y_validate)
                    
                    node_name.Fitting(x_train,y_train)
                
                if config.section_dict[key]['hyper_search'] == 'manual':
                    print 'HyperParameter Optimization for {}'.format(key)
                    node_name = Classifier(clf_type=config.section_dict[key]['classifier'],clf_opt=eval(config.section_dict[key]['optimization']),num_iter=eval(config.section_dict[key]['num_iter']))
                    node_name.Build(config.section_dict[key]['hyper_search'],eval(config.section_dict[key]['param']),None,None)
                    node_name.Fitting(x_processed,y_data)
                    
                if config.section_dict[key]['hyper_search'] == 'none':
                    if config.section_dict[key]['optimization'] == 'True':
                        x_train,x_validate,y_train,y_validate = train_test_split(x_processed,y_data,
                                                                     train_size = 0.75,
                                                                     test_size = 0.25)
                        node_name = Classifier(clf_type=config.section_dict[key]['classifier'],clf_opt=eval(config.section_dict[key]['optimization']),num_iter=eval(config.section_dict[key]['num_iter']))
                        node_name.Build(config.section_dict[key]['hyper_search'],eval(config.section_dict[key]['param']),x_validate,y_validate)
                        node_name.Fitting(x_train,y_train)
                    else:
                        node_name = Classifier(clf_type=config.section_dict[key]['classifier'],clf_opt=eval(config.section_dict[key]['optimization']),num_iter=eval(config.section_dict[key]['num_iter']))
                        node_name.Build(config.section_dict[key]['hyper_search'],eval(config.section_dict[key]['param']),None,None)
                        node_name.Fitting(x_processed,y_data)
                classifier_dict.update({key:node_name})
                classifier_list.append(key)
        self.classifier_dict = classifier_dict
        self.classifier_list = classifier_list
        
    def Set_Node_Outputs(self):    
        results_actual = []  
        print 'Setting Node Outputs'              
        for clf in reversed(sorted(self.classifier_list)):
            clf_dict = eval(self.config.section_dict[clf]['branch_dict'])
            if clf != 'DATA':
                for val,item in clf_dict.iteritems():
                    if self.config.section_dict[clf]['terminal'] == str(False):
                        clf_dict[val][0] = self.classifier_dict[clf_dict[val][0]]
                        
                    results_actual.append(item[1])                
            
            self.classifier_dict[clf].Set_Node(clf_dict,eval(self.config.section_dict[clf]['terminal']))
        self.results_actual = results_actual
        
    def Make_Virgin_Data_Sets(self):        
        print 'Making Virgin Data Sets'
        test_data_list = []
        y_data_list = []
        total_num = 0
        event_labels_dict = eval(self.config.section_dict['DATA']['event_labels'])
        for val in event_labels_dict.keys():
            if val == 0:
                x_virgin = self.data_set_class.Make_Virgin_Seismic() 
                #x_virgin_processed = self.data_process.Process(x_virgin,1)
                y_virgin = 0
                self.seismic_num = x_virgin.shape[0]
            if val == 1:
                x_virgin = self.data_set_class.Make_Virgin_Explosion()
                #x_virgin_processed = self.data_process.Process(x_virgin,1)
                y_virgin = 1
                self.exp_num = x_virgin.shape[0]
            if val == 2:
                x_virgin = self.data_set_class.Make_Virgin_Nuclear()
                #x_virgin_processed = self.data_process.Process(x_virgin,1)
                y_virgin = 2
                self.nuc_num = x_virgin.shape[0]
            test_data_list.append(x_virgin)
            y_data_list.append(y_virgin)
            total_num += len(x_virgin)       
        
        return test_data_list,y_data_list
    
    def Predict(self,data):
        data_processed,node = self.classifier_dict['DATA'].Process(data,0)
        return self.classifier_dict[node].Predicting(data_processed)
