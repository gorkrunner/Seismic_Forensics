#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 10:52:02 2017

@author: panderso

Purpose:  
        The purpose of this program is to pull information from the config
        file given to the program.  

Input:
    location: location of the configure file
    
Output:
    section_dict: dictionary with the lines from the configure file for use 
                    in the other classes of the program


"""

import ConfigParser
import io


class Configure_System(object):
    
    def __init__(self,location):
        self.location = location
        self.config = ConfigParser.RawConfigParser(allow_no_value=True)
        #self.Configure_Pull = Configure_Pull()
        
    def Configure_Pull(self):
        section_dict = {}
        with open(self.location) as f:
            config_f = f.read()
        self.config.readfp(io.BytesIO(config_f))
        self.sections = self.config.sections()
        for section in self.sections:
            section_dict.setdefault(section,{})
            for option in self.config.options(section):
                item = self.config.get(section,option)
                section_dict[section][option] = item
                
        self.section_dict = section_dict
                
            
if __name__=="__main__":
   
    pass 
    #configure = Configure_System('config_classifier.ini')
    #configure.Configure_Pull()
    #print configure.location
    #print configure.config
    #print configure.sections
    #print ' '
    #print configure.section_dict
    
    
