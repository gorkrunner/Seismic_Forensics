#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 12:32:43 2017

@author: panderso

Purpose:
    The purpose of this class is to build the data set from numpy files.
    
    
Input:
    Location of numpy files for each type of data set
    
Output:
    data_set: data set following sizing of data.
                data for machine learning purposes must be of equal or very close
                sizes to ensure accurate representation is availble during
                data set splitting procedures



"""

import numpy as np

class Data_Set(object):
    
    def __init__(self,locations_of_files):
        '''
        Locations of Files will be given in a list [seismic,chemical,nuclear]
        '''
        self.locations_of_files = locations_of_files
        
        self.Pull_Data_from_File()
        
    def Pull_Data_from_File(self):
        '''
        Pull data from associated locations
        '''
        
        
        self.seismic_data_raw = np.load(self.locations_of_files[0])
        self.explosion_data_raw = np.load(self.locations_of_files[1])
        self.nuclear_data_raw = np.load(self.locations_of_files[2])
        
        self.seismic_data_raw = self.Remove_Bad_Rows(self.seismic_data_raw)
        self.explosion_data_raw = self.Remove_Bad_Rows(self.explosion_data_raw)
        self.nuclear_data_raw = self.Remove_Bad_Rows(self.nuclear_data_raw)
        
        
        self.test_size = .3  #Size of virgin test set

        test_num = np.array((len(self.seismic_data_raw),
                             len(self.explosion_data_raw),
                                len(self.nuclear_data_raw)))

        min_num_test = int(np.min(test_num)*self.test_size)
        
        num_test_seismic = np.arange(0,test_num[0],1)
        num_test_exp = np.arange(0,test_num[1],1)
        num_test_nuc= np.arange(0,test_num[2],1)
        
        self.test_set_row_seismic = np.random.choice(num_test_seismic,min_num_test,replace=False)
        self.test_set_row_exp = np.random.choice(num_test_exp,min_num_test,replace=False)
        self.test_set_row_nuc = np.random.choice(num_test_nuc,min_num_test,replace=False)

        self.prob_matrix_seismic = np.ones((test_num[0]),dtype=np.float64)
        self.prob_matrix_exp = np.ones((test_num[1]),dtype=np.float64)
        self.prob_matrix_nuc = np.ones((test_num[2]),dtype=np.float64)
        
        self.prob_matrix_seismic[self.test_set_row_seismic.astype(int)] = 1e-50
        self.prob_matrix_exp[self.test_set_row_exp.astype(int)] = 1e-50
        self.prob_matrix_nuc[self.test_set_row_nuc.astype(int)] = 1e-50

        self.prob_matrix_seismic = self.prob_matrix_seismic / float(np.sum(self.prob_matrix_seismic))
        self.prob_matrix_exp = self.prob_matrix_exp / float(np.sum(self.prob_matrix_exp))
        self.prob_matrix_nuc = self.prob_matrix_nuc / float(np.sum(self.prob_matrix_nuc))
              
    def Remove_Bad_Rows(self,data):

        bad_rows_inf = np.where(np.isinf(data)==True)[0]
        bad_rows_nan = np.where(np.isnan(data)==True)[0]
        
        data = np.delete(data,bad_rows_inf,axis=0)
        data = np.delete(data,bad_rows_nan,axis=0)
        
        return data
                    
        
    def Make_Data_Set(self,percentage_of_events_labels):
        
        '''
        percentage_of_events_labels: gives you the percentage of the finished
            array for each event type and the label associated with it
            must be a list of lists***
        '''            
       
        
        data_raw_array = np.array((len(self.seismic_data_raw),len(self.explosion_data_raw),len(self.nuclear_data_raw)))
        
        num_min_events = []
        for i,percentage in enumerate(percentage_of_events_labels):
            if percentage[0] != 0:
                num_min_events.append((data_raw_array[i] / float(percentage[0])))
                            
        min_size_of_array = int(np.min(num_min_events))
      
        num_rows_seismic = int(min_size_of_array * percentage_of_events_labels[0][0])
        num_rows_explosion = int(min_size_of_array * percentage_of_events_labels[1][0])
        num_rows_nuclear = int(min_size_of_array * percentage_of_events_labels[2][0])
        
        seismic_rows_random = np.random.choice(data_raw_array[0],num_rows_seismic,replace=False,p = self.prob_matrix_seismic)
        exp_rows_random = np.random.choice(data_raw_array[1],num_rows_explosion,replace=False,p = self.prob_matrix_exp)
        nuc_rows_random = np.random.choice(data_raw_array[2],num_rows_nuclear,replace=False, p = self.prob_matrix_nuc)
    
        x_data_set = np.vstack((self.seismic_data_raw[seismic_rows_random,:],
                              self.explosion_data_raw[exp_rows_random,:]))
        x_data_set = np.vstack((x_data_set,self.nuclear_data_raw[nuc_rows_random,:]))
        
        
        y_seismic = np.zeros((1,len(self.seismic_data_raw[seismic_rows_random,:])))
        y_seismic[:] = percentage_of_events_labels[0][1]
        
        y_exp = np.zeros((1,len(self.explosion_data_raw[exp_rows_random,:])))
        y_exp[:] = percentage_of_events_labels[1][1]
        
        y_nuc = np.zeros((1,len(self.nuclear_data_raw[nuc_rows_random,:])))
        y_nuc[:] = percentage_of_events_labels[2][1] 
        
        y_data_set = np.hstack((y_seismic,y_exp))
        y_data_set = np.hstack((y_data_set,y_nuc))
                         
        data_set = np.hstack((x_data_set,y_data_set.T))
        
        return data_set
    
    def Make_Virgin_Seismic(self):
        return self.seismic_data_raw[self.test_set_row_seismic,:]
    def Make_Virgin_Explosion(self):
        return self.explosion_data_raw[self.test_set_row_exp,:]
    def Make_Virgin_Nuclear(self):
        return self.nuclear_data_raw[self.test_set_row_nuc,:]
        
    
if __name__=="__main__":
    pass
    #location_of_data = ['/home/ENP/gne18m/panderso/Desktop/Data/Numpy/40HZ_16384_PADDED/Earthquake_array.npy',
    #                '/home/ENP/gne18m/panderso/Desktop/Data/Numpy/40HZ_16384_PADDED/Explosion_array.npy',
    #                '/home/ENP/gne18m/panderso/Desktop/Data/Numpy/40HZ_16384_PADDED/Nuclear_array.npy']
    
    #data_set_class = Data_Set(location_of_data) 
    
    #percentage_list = [[0.5,0],[0.25,1],[0.25,1]]
    
    #node_0_data = data_set_class.Make_Data_Set(percentage_list) 
