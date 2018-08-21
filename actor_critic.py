# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 10:45:50 2017

@author: njp60
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import time 
import gym 
import tflearn


class ActorCritic():
    def __init__(self,state_size,action_size,agent_scope, global_optimizer):
           
     # for each each we have a "Local" agent scope plus an agent with global scope   
     with tf.variable_scope(agent_scope): 
                
        self.state_size=state_size
        self.action_size=action_size
        self.global_optimizer=global_optimizer
 
       # place holders for inputs
        self.state=tf.placeholder(shape=[None,state_size], name='state',dtype='float32')
        
       #==============================================================================
       #         Output of the neural networks        
       #==============================================================================
        self.policy_output=self.create_policy_network()
        self.value_output=self.create_value_network()
        
        #==============================================================================
        # Create the place holders for actions and target to train individual agents
        # Create loss function
        # Obtain the gradients       
        #==============================================================================
        if agent_scope !='global' :       
            
            """
            1. we compute the gradients for the "local" agent network
            2. Create the ops for updating the "global" network           
            """    
            
            # place holder for selected action and target (v/Q/TD error) 
            # Places holders                            
            self.target_v=tf.placeholder(shape=[None,1], name='target', dtype='float32')
            self.action=tf.placeholder(shape=[None,action_size], name='action', dtype='float32')
            self.advantage=tf.placeholder(shape=None,dtype='float32') 
            
            
            # Policy loss            
            self.action_onehot=tf.one_hot(self.action,self.action_size, dtype='float32')
            self.selected_action=tf.reduce_sum((self.policy_output*self.action_onehot),[1])
            self.entropy=-tf.reduce_sum(self.policy_output*tf.log(self.policy_output))
            self.policy_loss=-tf.reduce_sum(tf.log(self.selected_action)*self.advantage)
            
            
            # Value Loss
            self.value_loss=tf.square_difference(self.value_output,self.target_v)
    
            
            # Total loss
            self.total_loss=0.5*self.value_loss+self.policy_loss-self.entropy*0.01
            
           
            # Local gradients and train         
            local_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,agent_scope)
            self.local_gradients=tf.gradients(self.total_loss,local_vars)  # this is gradients between total loss and local variables
            self.var_norms=tf.global_norm(local_vars)
            grads,self.grad_norms=tf.clip_by_global_norm(self.local_gradients,40.0)

                          
  
            global_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'global')
            
            # Apply local gradients Gradients to global gradients 
            # We need apply the gradients to the global optimizer
            self.apply_grads=self.global_optimizer.apply_gradients(zip(grads,global_vars))                  
                  
                          
    
    
    
    def create_policy_network(self): 
        """       
        A ff network with two hidden layers with input as state dim and output as action probablities  
        """
               #self.state_size
        output_dim=self.action_size
        hidden1_dim=100
        hidden2_dim=200
        
        fc1=tflearn.fully_connected(self.state,hidden1_dim)
        fc2=tflearn.fully_connected(fc1,hidden2_dim)
        action_probablities=tflearn.fully_connected(fc2,output_dim,activation='softmax')
    
        return action_probablities
    
    def create_value_network(self):  
        """       
        A ff network with two hidden layers with input as state dim and output as action probablities  
        """
        output_dim=1 #we are only interested in the output target
        
        hidden1_dim=100
        hidden2_dim=200
        
        fc1=tflearn.fully_connected(self.state,hidden1_dim)
        fc2=tflearn.fully_connected(fc1,hidden2_dim)
        action_probablities=tflearn.fully_connected(fc2,output_dim,activation='relu')
    
        return action_probablities 

 
    