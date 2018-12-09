#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 13:46:07 2017

@author: panderso

Purpose:
    The purpose of this class is to do the data processing within the data
    processing node.  The operations performed in the node is set by the user
    inputted configure file.  
    
Input:
    FFT: Different modes of FFT:
            Full Fourier Transform
            Welch Method using boxcar window
            None
    Detrend: Which detrending technique is used
            Constant
            Linear
            None
    Opt: Optimization routine to determine best set of options
    Freq: frequency of the data points collected in Hz.  Set to 40

Output:
    data: data after data processing techniques have been applied




"""

import numpy as np
from scipy.signal import welch,detrend,butter,lfilter
from sklearn.preprocessing import normalize
from scipy.optimize import curve_fit
from Multi_Node_Class import Multi_Node
import sys


class Data_Processing(Multi_Node):
    
    def __init__(self,FFT='Full',Detrend='linear',Norm='none',opt=False,freq = 40):
        
        self.FFT = FFT
        self.Detrend = Detrend
        self.Norm = Norm
        self.opt = opt
        self.freq = freq
        self.Multi_Node = Multi_Node()
        
    #Create the node with out branches
    def Set_Node(self,Out_Branch,Terminal):
        self.Multi_Node.Out_Branch = Out_Branch
        self.Multi_Node.Terminal = Terminal  
    
#    def Base_2(self,x):
#        try:
#            n_inside = x.shape[1]
#            a=1
#            #n = np.log2(x.shape[1])
#        except:
#            n_inside = x.shape[0]
#            a=0
#            #n = np.log2(x.shape[0])
#        n = np.log2(n_inside)
#        n_ceil = int(np.ceil(n))
#        num = 2**n_ceil
#        num_req = num-n_inside
#        
#        if a == 1:
#            add = np.zeros((x.shape[0],num_req))
#        else:
#            add = np.zeros((num_req,))
#        x_new = np.append(x,add,axis=a)
#        return x_new  
    
    def Sine_Linear_Func(self,x,a,b,c,d,e):
        return d*np.sin(a*x+e)+b*x+c
    def Sine_Func(self,x,a):
        return np.sin(a*x)
    def RMS_Calc(self,y_new):
        return np.sqrt(np.mean(np.square(y_new-self.data)))
    
    def Pad_Array(self,data):
        
        max_len=16384
        
        if data.ndim == 1:
            data = np.reshape(data,(1,data.shape[0]))
            
        data_pad = np.array([np.pad(row,(0,max_len-len(row)),mode='constant') for row in data])
        data_pad = np.reshape(data_pad,(max_len,))
        return data_pad
    
    def Detrend_Process(self,data,axis):
        if self.opt == False or self.opt == 'False':
            if self.Detrend.upper() == 'LINEAR' or self.Detrend.upper() == 'CONSTANT':
                data = detrend(data,axis=axis,type=self.Detrend.lower())
            elif self.Detrend == 'none' or self.Detrend == None:
                pass 
            elif type(eval(self.Detrend)) == float:
                if data.ndim == 2:
                    shape_val = 1
                else:
                    shape_val=0
                bp_new = np.arange(0,data.shape[shape_val],int(data.shape[shape_val]*eval(self.Detrend)))
                data = detrend(data,axis=shape_val,type='linear',bp=bp_new)
            else:
                raise IOError('Detrend Input is invalid (linear,constant,none)')
        return data        
        #self.Norm_Process(data,axis)
    
    def Norm_Process(self,data,axis):
        if self.Norm.upper() == 'L1' or self.Norm.upper() == 'L2' or self.Norm.upper() == 'MAX':
                data = data.reshape(1,-1)
                data = normalize(data,norm=self.Norm.lower(),axis=axis)
        elif self.Norm == 'none' or self.Norm == None:
                data = data
        else:
                raise IOError('Norm Input is invalid (l1,l2,max,none)')
            
        return data
        #self.FFT_Process(data,axis)
    
    def FFT_Process(self,data,axis):
        
        if self.FFT.upper() == 'FULL':
                #data = self.Base_2(data)
                data = np.abs(np.fft.rfft(data,n=16384))  
        elif self.FFT.upper() == 'WELCH':
                #data = self.Base_2(data)
                try:
                    data_new = []
                    if data.ndim > 1:
                        for row in data:
                            data_welch,_ = welch(row,window='boxcar',nfft=16384)
                            data_new.append(data_welch)
                        data = np.asarray(data_new)
                    else:
                        data,_ = welch(data,window='boxcar')
                except ValueError as err:
                    print err
        elif self.FFT.upper() == 'NONE' or self.FFT == None:
                data = self.Pad_Array(data)
                return data
        else:
                raise IOError('FFT Input is invalid (Full, Welch, None)')  
        
        return data
    
    def Process(self,data,axis):       
        
        if data.ndim == 1:
            shape_array = (1,data.shape[0])
            data = np.reshape(data,shape_array)
            
        for i,row in enumerate(data):
            last_nonzero = np.nonzero(row)[0][-1]
            row_new = self.Detrend_Process(row[:last_nonzero],0)
            row_new = self.Norm_Process(row_new,0)
            if row_new.ndim == 2:
                row_new = np.reshape(row_new,(row_new.shape[1],))
            row_new = self.FFT_Process(row_new,0)
            
            if i == 0:
                data_array = np.zeros((data.shape[0],len(row_new))) 
                
            data_array[i,:] = row_new
        
        if self.Multi_Node.Out_Branch != None:
            return data_array,self.Multi_Node.Out_Branch[0][0]
        else:
            return data_array
        '''
	
        if self.opt == False or self.opt == 'False':
            if self.Detrend.upper() == 'LINEAR' or self.Detrend.upper() == 'CONSTANT':
                data = detrend(data,axis=axis,type=self.Detrend.lower())
            elif self.Detrend == 'none' or self.Detrend == None:
                pass 
            elif type(eval(self.Detrend)) == int:
                if data.ndim == 2:
                    shape_val = 1
                else:
                    shape_val=0
                bp_new = np.arange(0,data.shape[shape_val],eval(self.Detrend))
                data = detrend(data,axis=shape_val,type='linear',bp=bp_new)
            else:
                raise IOError('Detrend Input is invalid (linear,constant,none)')
    
             
            if self.Norm.upper() == 'L1' or self.Norm.upper() == 'L2' or self.Norm.upper() == 'MAX':
                data = data.reshape(1,-1)
                data = normalize(data,norm=self.Norm.lower(),axis=axis)
            elif self.Norm == 'none' or self.Norm == None:
                pass
            else:
                raise IOError('Norm Input is invalid (l1,l2,max,none)')
            
                
            if self.FFT.upper() == 'FULL':
                data = self.Base_2(data)
                data = np.abs(np.fft.rfft(data,n=16384))  
            elif self.FFT.upper() == 'WELCH':
                data = self.Base_2(data)
                try:
                    data_new = []
                    if data.ndim > 1:
                        for row in data:
                            data_welch,_ = welch(row,window='boxcar',nfft=16384)
                            data_new.append(data_welch)
                        data = np.asarray(data_new)
                    else:
                        data,_ = welch(data,window='boxcar')
                except ValueError as err:
                    print err
            elif self.FFT.upper() == 'NONE' or self.FFT == None:
                pass
            else:
                raise IOError('FFT Input is invalid (Full, Welch, None)')  
                
            if self.Multi_Node.Out_Branch != None:
                return data,self.Multi_Node.Out_Branch[0][0]
            else:
                return data
                
        elif self.opt == True or self.opt == 'True':
            return self.Optimize()                          
                          
        '''
    
    def Optimize(self):
        
        if self.data.ndim > 1:
            y_final = np.zeros((self.data.shape[0],self.data.shape[1]))
        else:
            shape_array = ((1,self.data.shape[0]))
            y_final = np.zeros((self.data.shape[0],))
            self.data = np.reshape(self.data,shape_array)
            last_nonzero = np.nonzero(self.data)[1][-1]
            self.data = self.data[:,:last_nonzero]
        
        x_data = np.linspace(0,len(self.data)/float(self.freq),self.data.shape[1])
        
        #plt.plot(x_data,self.data,label='Original Data')
        
        for i,row in enumerate(self.data):
            shape_array = ((1,self.data.shape[1]))
            row = np.reshape(row,shape_array)
           
            
            #try:
            y_constant = detrend(row,type='constant')
            y_constant = np.reshape(y_constant,shape_array)
            RMS_constant = self.RMS_Calc(y_constant)
#            except:
#                RMS_constant = np.inf
#                y_constant = []
#                pass
            
            #try:
            y_linear = detrend(row,type='linear')
            y_linear = np.reshape(y_linear,shape_array)
            RMS_linear = self.RMS_Calc(y_linear)
#            except:
#                y_linear = []
#                RMS_linear = np.inf
#                pass
#            
            #try:
                
            popt_sine,pcov_sine = curve_fit(self.Sine_Func,x_data,row[0,:],bounds=(-2*np.pi,2*np.pi))
            y_sine = self.Sine_Func(x_data,*popt_sine)
            y_sine = np.reshape(y_sine,shape_array)
            y_sine = row-y_sine
            RMS_Sine = self.RMS_Calc(y_sine)
#            except:
#                y_sine = []
#                RMS_Sine = np.inf
#                pass
        
            #try:
            param_bounds = [[-2*np.pi,-np.inf,-np.inf,-np.inf,-np.pi],[2*np.pi,np.inf,np.inf,np.inf,np.pi]]
            popt_sine_linear,pcov_sine_linear = curve_fit(self.Sine_Linear_Func,x_data,row[0,:],bounds=param_bounds)
            y_sine_linear = self.Sine_Linear_Func(x_data,*popt_sine_linear)
            y_sine_linear = np.reshape(y_sine_linear,shape_array)
            y_sine_linear = row-y_sine_linear
            RMS_Sine_Linear = self.RMS_Calc(y_sine_linear)
#            except:
#                y_sine_linear = []
#                RMS_Sine_Linear = np.inf
#                pass
            
                           
            y_list = [y_constant,y_linear,y_sine,y_sine_linear]
            RMS_array = np.array((RMS_constant,RMS_linear,RMS_Sine,RMS_Sine_Linear))
            min_RMS_arg = np.argmin(RMS_array)
            
            if self.data.shape[0] > 1:
                y_final[i,:] = y_list[min_RMS_arg]
            else:
                y_final = y_list[min_RMS_arg]
                
                '''Need to repad to 16384
                '''
        
        
        if self.Multi_Node.Out_Branch != None:
            return y_final,self.Multi_Node.Out_Branch[0][0]
        else:
            return y_final
        
    
if __name__=="__main__":
    pass 
