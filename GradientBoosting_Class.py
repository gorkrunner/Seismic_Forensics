#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
		Peter Anderson
		Thesis Work

Purpose: 
	Create Classifier class to be used in conjunction with multi_node
	to create nodes of binary classification tree.  Can be also used 
	to build a multiclassifier for an individual node.
"""

#Import important functions
import sklearn.ensemble
import sklearn
from sklearn.svm import OneClassSVM
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV,cross_val_score
import numpy as np
from scipy.stats import randint as sp_randit
from sklearn.model_selection import train_test_split
#from Node_Class import Node
from Multi_Node_Class import Multi_Node
import time
import pickle
from scipy.stats import norm as sp_norm
from scipy.stats import halfnorm as sp_halfnorm
#from simulated_annealing.optimize import SimulatedAnneal
#from mlxtend.classifier import StackingClassifier
import pandas as pd
import multiprocessing as mp
import os

#Establish Classifier Class
class Classifier(Multi_Node):
    
    #Define some initial variables inputted by user
    def __init__(self,clf_type = 'GradientBoosting',num_iter = '10',clf_opt=False):
        self.clf_type = clf_type	#What type of classifier for the node
        self.num_iter = num_iter	#If randomized hyperparameter search number of times random number is pulled
        self.Multi_Node = Multi_Node()	#Create node function from Multi_Node
        self.Optimization = clf_opt
                                    
    #Create the node with out branches
    def Set_Node(self,Out_Branch,Terminal):
        self.Multi_Node.Out_Branch = Out_Branch
        self.Multi_Node.Terminal = Terminal        

    #Build the classifier (train, hyperparameters)                
    def Build(self,clf_hyper_search,hyper_params,xvalidate,yvalidate):        
        
        self.xvalidate = xvalidate
        self.yvalidate = yvalidate
        
        self.search = clf_hyper_search
        
        if self.Optimization == False:
    	#Determine the type of hyperparameter search
            if clf_hyper_search == 'random':
                print 'Random Search'
                self.Random_Search(hyper_params)
            if clf_hyper_search == 'grid':
                self.Grid_Search(hyper_params)
            if clf_hyper_search == 'none' or clf_hyper_search == None:
                if self.clf_type == 'GradientBoosting':
                    self.classifier = sklearn.ensemble.GradientBoostingClassifier(verbose=2.0)
                if self.clf_type == 'RandomForest':
                    self.classifier = sklearn.ensemble.RandomForestClassifier()
                if self.clf_type == 'OneClassSVM':
                    self.classifier = OneClassSVM(nu=0.1,kernel='sigmoid')
            if clf_hyper_search == 'manual':
                #print 'Manual Settings'
                
                if self.clf_type == 'GradientBoosting':
                    hyper_params.update({'verbose':2.0})
                    self.classifier = sklearn.ensemble.GradientBoostingClassifier(**hyper_params)
                if self.clf_type == 'RandomForest':
                    self.classifier = sklearn.ensemble.RandomForestClassifier(**hyper_params)
            if clf_hyper_search == 'annealing':
                print 'Annealing Search'
                self.Annealing_Search(hyper_params)
         
        if self.Optimization == True:
            self.Classifier_Optimization()
        
         
                
  #  def Annealing_Search(self,hyper_params):
        
  #      if self.clf_type == 'GradientBoosting':
  #          clf = sklearn.ensemble.GradientBoostingClassifier(verbose=2.0)
  #      elif self.clf_type == 'RandomForest':
  #          clf = sklearn.ensemble.RandomForestClassifier()
  #      else:
  #          print 'Classifier not in class...GradientBoosting chosen'
  #          clf = sklearn.ensemble.GradientBoostingClassifier()
            
            
  #      sa = SimulatedAnneal(clf,hyper_params,T=100,T_min=0.01,alpha=0.75,verbose=True,
  #                   max_iter=0.25,n_trans=10,max_runtime=750,cv=3,
  #                   scoring='f1_macro',refit=True,n_jobs=10)  #Set to 10 due to memory issues (mp.cpu_count())
        
  #      sa.fit(self.xvalidate,self.yvalidate)
        
  #      self.classifier = sa.best_estimator_

   #     self.Save_Hyper_Optimization(sa.best_params_)
        
    def Save_Hyper_Optimization(self,results_dict):
        
        filename = 'Hyperparameter_Optimization.csv'
        df_new = pd.DataFrame.from_dict(results_dict)
        if os.path.exists(filename):
            df = pd.read_csv(filename,sep=',')
            df_combined =   df.append(df_new)
        else:
            df_combined = df_new
        
        df_combined.to_csv(filename)

    #Randomized Search
    def Random_Search(self,hyper_params):
        
        if type(hyper_params) == dict:        
            if self.clf_type == 'GradientBoosting':
                model = sklearn.ensemble.GradientBoostingClassifier(verbose=2.0)
                rsearch = RandomizedSearchCV(model,param_distributions=hyper_params,n_iter = int(self.num_iter),n_jobs=-1,verbose=2,pre_dispatch='n_jobs')
                rsearch.fit(self.xvalidate,self.yvalidate)
                self.classifier = sklearn.ensemble.GradientBoostingClassifier(**rsearch.best_params_)
                self.Save_Hyper_Optimization(rsearch.cv_results_)
                print 'GradientBoosting Optimization Complete best score of {}'.format(str(rsearch.best_score_))
                for key,value in rsearch.best_params_.iteritems():
                    print 'Hyper {} with value of {}'.format(str(key),str(value))
                
            if self.clf_type == 'RandomForest':
                model = sklearn.ensemble.RandomForestClassifier()
                rsearch = RandomizedSearchCV(model,param_distributions=hyper_params,n_iter = self.num_iter,n_jobs=-1,verbose=2,pre_dispatch='n_jobs')
                rsearch.fit(self.xvalidate,self.yvalidate)
                self.classifier = sklearn.ensemble.RandomForestClassifier(**rsearch.best_params_)
                self.Save_Hyper_Optimization(rsearch.cv_results_)
                print 'RandomForest Optimization Complete best score of {}'.format(str(rsearch.best_score_))
                for key,value in rsearch.best_params_.iteritems():
                    print 'Hyper {} with value of {}'.format(str(key),str(value))
        elif type(hyper_params) == list:
            if self.clf_type == 'GradientBoosting':
                dict_rsearch = {}
                for i,hyper_dict in enumerate(hyper_params):
                    print 'Optimizing {}'.format(str(hyper_params[i].keys()[0]))
                    if i == 0:
                        model = sklearn.ensemble.GradientBoostingClassifier(verbose=2.0)
                    else:
                        dict_rsearch.update({'verbose':2.0})
                        model = sklearn.ensemble.GradientBoostingClassifier(**dict_rsearch)
                    rsearch =  RandomizedSearchCV(model,param_distributions=hyper_dict,n_iter = int(self.num_iter),n_jobs=-1,verbose=2,pre_dispatch='n_jobs')
                    rsearch.fit(self.xvalidate,self.yvalidate)
                    dict_rsearch.update(rsearch.best_params_)
                    print 'Optimized {} with value of {}'.format(str(hyper_params[i].keys()[0]),rsearch.best_params_[hyper_params[i].keys()[0]])
                self.classifier = sklearn.ensemble.GradientBoostingClassifier(**dict_rsearch)
                self.Save_Hyper_Optimization(rsearch.cv_results_)
        else:
            raise IOError('Wrong data type for hyper_params.  Dictionary only')
            
            
    #Grid Search for hyperparamter
    def Grid_Search(self,hyper_params):

        if type(hyper_params) == dict:
            if self.clf_type == 'GradientBoosting':
                model = sklearn.ensemble.GradientBoostingClassifier(verbose=2.0)
                gsearch = GridSearchCV(model,hyper_params,verbose=2,n_jobs=-1,pre_dispatch='n_jobs')
                gsearch.fit(self.xvalidate,self.yvalidate)
                self.classifier = sklearn.ensemble.GradientBoostingClassifier(**gsearch.best_params_)     
                self.Save_Hyper_Optimization(gsearch.cv_results_)
                print 'GradientBoosting Optimization Complete'
                for key,value in gsearch.best_params_.iteritems():
                    print 'Hyper {} with value of {}'.format(str(key),str(value))
            if self.clf_type == 'RandomForest':
                model = sklearn.ensemble.RandomForestClassifier()
                gsearch = GridSearchCV(model,hyper_params,verbose=2,n_jobs=-1,pre_dispatch='n_jobs')
                gsearch.fit(self.xvalidate,self.yvalidate)                
                self.classifier = sklearn.ensemble.GradientBoostingClassifier(**gsearch.best_params_)
                self.Save_Hyper_Optimization(gsearch.cv_results_)
                print 'RandomForest Optimization Complete'
                for key,value in gsearch.best_params_.iteritems():
                    print 'Hyper {} with value of {}'.format(str(key),str(value))
        else:
            raise IOError('Wrong data type for hyper_params.  Dictionary Only')  

            
    def Classifier_Optimization(self):
        print 'Classifier Optimization Selected for Node'
        
        clf1 = sklearn.ensemble.GradientBoostingClassifier()
        clf1_hypers = {'n_estimators':sp_randit(100,500)}
        clf2 = sklearn.ensemble.RandomForestClassifier()
        clf2_hypers = {'n_estimators':sp_randit(10,100)}
        clf3 = sklearn.ensemble.AdaBoostClassifier()
        clf3_hypers = {'n_estimators':sp_randit(10,100)}
        clf4 = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)
        clf4_hypers = None
        
        clf_old = [clf1,clf2,clf3,clf4]
        clf_names = ['Gradient','Random','AdaBoost','KNeighbors','Stacked']
        clf_hypers = [clf1_hypers,clf2_hypers,clf3_hypers,clf4_hypers]
        
        print 'Optimizing Hyperparameters'
        clf_list = []
        i=0
        for clf,hyper_params in zip(clf_old,clf_hypers):
            if hyper_params!=None:
                print 'Optimizing {} Classifier'.format(clf_names[i])
                rsearch = RandomizedSearchCV(clf,param_distributions=hyper_params,n_iter = int(self.num_iter),n_jobs=-1,verbose=2,pre_dispatch='n_jobs')
                rsearch.fit(self.xvalidate,self.yvalidate)
                clf_new = sklearn.ensemble.GradientBoostingClassifier(**rsearch.best_params_)
                clf_list.append(clf_new)
            else:
                clf_list.append(clf)
            i+=1
            clf_list.append(clf)
 #       sclf = StackingClassifier(classifiers=[clf1,clf2,clf3,clf4],use_probas=True,average_probas=False,meta_classifier=clf4)        
 #       clf_list.append(sclf)

        i=0
        scores_mean = np.zeros((len(clf_list)))
        print 'Determining Optimal Classifier'
        for clf,label in zip(clf_list,clf_names):
            scores = cross_val_score(clf,self.xvalidate,self.yvalidate,cv=10, scoring='f1_macro',n_jobs=-1,verbose=2,pre_dispatch='n_jobs')
            scores_mean[i] = scores.mean()
            i+=1
        max_score_arg = np.argmax(scores_mean)
        
        print 'Best Classifier is {} with a score of {:4.2f}'.format(clf_names[max_score_arg],scores_mean[max_score_arg])
        
        self.classifier = clf_list[max_score_arg]
        
            
    #Following hyperparamter optimization fitting the classifier to the training data
    def Fitting(self,xtrain,ytrain):
        if self.clf_type != 'OneClassSVM':
            self.classifier.fit(xtrain,ytrain)
        else:
            self.classifier.fit(xtrain)
        
    #Predict the outcome of the classifier...if the nodes are built it will run through all of them
    def Predicting(self,xtest):   
        
        x_reshape = xtest.reshape(1,-1)
        self.classifier_predict = int(self.classifier.predict(x_reshape))
        
        
        if self.clf_type == 'OneClassSVM':
            confidence_max = 0.0
        else:
            confidence = self.classifier.predict_proba(x_reshape)
            confidence_max = np.max(confidence)
                   
        
        #Need to be careful of shape of array coming into prediction
        if self.Multi_Node.Terminal != True:
            return self.Multi_Node.Out_Branch[self.classifier_predict][0].Predicting(x_reshape) + " " + self.Multi_Node.Out_Branch[self.classifier_predict][1] +' (%.2f) '%confidence_max
        else:
            return self.Multi_Node.Out_Branch[self.classifier_predict][1] +' (%.2f) '%confidence_max

#            
#            if events[int(y_test_validate[row])] == dic['Run %d'%i]:
#                num_correct+=1
#        num_correct_avg += num_correct
#        t_run[j] = t_end-t_start
#        print 'Loop %d' %j
#
#                
#    print 'Average Percentage Correct ', num_correct_avg / float(500*500)
#    print 'Average Time to Run ', np.average(t_run)
