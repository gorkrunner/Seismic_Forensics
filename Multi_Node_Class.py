#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:07:13 2017

@author: panderso

Purpose:
    The purpose of this class is to build the nodes which the classifiers and
    data processing section will be placed in.  
"""

import numpy as np

class Multi_Node(object):


    #Branches definition is a dictionary to allow for multi class
    def __init__(self,ID=None,Out_Branch=None,Terminal = False):
        self.ID = ID
        self.Out_Branch = Out_Branch
        self.Terminal = Terminal
        self.dic = {'Node '+str(self.ID):{}}
            
    def Process(self,data):
        
        if type(self.Out_Branch) == dict:
            if data >= 0.75:
                resp = 0
                return self.Out_Branch[str(resp)]
            if data >= 0.5 and data<0.75:
                resp = 1
                return self.Out_Branch[str(resp)]
            if data >= 0 and data < 0.5:
                resp = 2
                return self.Out_Branch[str(resp)]    
        else:
            raise IOError('Out_Branch is not a dictionary')
    
                
                
if __name__=="__main__":
    pass
