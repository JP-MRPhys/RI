# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 14:07:12 2017
@author: jehill
"""

import numpy as np
import time
import tensorflow as tf

from actor_critic import ActorCritic



#SHARE_TIMESTEP_COUNTER=0        # SHARED counter for MAX TIME STEP
#GLOBAL_MAX_TIME_STEP=8000000    # SHARED constant for max times across all agents
#MAX_TIMESTEP_PER_EPISODE=32     # we only train an episode to this time in each agent

class training_agent():
 
    def __init__(self,ids,env,session,global_optimizer, global_max_timesteps, state_size,action_size):
        
            self.ids=ids
            self.env=env
            
            self.agent_name = "agent_id_"+ str(ids)
            self.agent_scope ="agent_id_"+ str(ids)

            self.tf_session=session            
            self.state_size=state_size
            self.action_size=action_size
            
            self.local_agent=ActorCritic(self.state_size,self.action_size,self.agent_scope,global_optimizer) #create the AC network            
            self.initial_local_ops=self.swap_tf_ops('global', self.agent_scope) # get global ops to reset local agents 
            
            
            self.global_max_timesteps=global_max_timesteps
            self.MAX_TIMESTEP_PER_EPISODE=500
            self.buffer_length=10        # len of the buffer            
            self.gamma=0.999
            
     
    def run_agent(self):
    
        env=self.env
        session=self.tf_session

        state_size=self.state_size   
        action_size=self.action_size
        
        print ("Starting agent %s" %(self.agent_id))
        
        global SHARED_TIMESTEP_COUNTER  # do I need to initialise the variable? is the correct?
        

        with session.as_default(), session.graph.as_default():
             
           t=0 
           
           while SHARED_TIMESTEP_COUNTER<self.global_max_timestep:
      
        # assign the transfer the global variable (Theata,Thetav) to agent (Theta' and thetav')
        # Initialise the the gradient of the learner to global network i.e. apply weights for the global to local network    
        # Train while timestep counter which is share across all agents to MAXIMUM GLOBAL timestemp
                
                session.run(self.initial_local_ops)    
            
                state = env.reset()
                state=np.float32(state)
                terminal = False      
                episode_reward=0


                experiences=[]                
                t_start=t              # keep a track of the starting time of the episode
                    
            
                while(not terminal or (t-t_start==self.MAX_TIME_PER_EPISODE)): 
                    
                    #random sleep between one sec to improve the agent performance
                
                    #print ("Step 1 select an action")
                    action_prob,Q_value=self.predict_local_agent(np.reshape(state,[1,state_size]))
                    action=np.random.choice(action_prob,p=action_prob)
                    action=np.argmax(action,action_prob)
                               
            
                    state_next,reward,terminal,_=env.step(action) 
                    state_next=np.float32(state_next)
                
                    action_prob_next,Q_value_next=self.predict_local_agent(np.reshape(state_next,[1,state_size]))
                    action_next=np.random.choice(action_prob_next,p=action_prob_next)
                    action_next=np.argmax(action_next,action_prob_next)

                    # collect the rollouts
                    experiences.append([state,action,reward,state_next,terminal])
                    
                    episode_reward+=reward
                    state=state_next    
                    
                    # Update the counters 
                    SHARED_TIMESTEP_COUNTER +=1                    
                    t +=1

                    
                    # and we train here if the expeience buffer is full and episode is not done
                    # train the local agent and apply it to global agent
                    if len(experiences)==self.buffer_length and not terminal:
                        self.update_global_network(experiences)
                        experiences=[] # reset the buffer
                             
 
                if terminal :
                       print("Global time step:s %s reward: %s and agentid:%s " %(SHARED_TIMESTEP_COUNTER, episode_reward, self.agent_id))  
                       
                
                # the episode ended and we train 
                # train the local agent and apply it to global agent
                if len(experiences)> 0: 
                            self.update_global_network(experiences)
                            experiences=[] # reset the buffer
                        
                  
    def predict_local_agent(self,state):
        """            
        Obtain the local "action probability" and "q value" for every input "state" for the local network
        
        """
        feed_dict={self.actor_critic.state: state}
                  
        action_prob,q_value =self.session.run([self.local_agent.policy_output,
                                               self.local_agent.value_output], 
                                               feed_dict=feed_dict)          
        
        return action_prob, q_value

    
    def predict_local_Q_value(self,state):
        """            
        Obtain the local "action probability" and "q value" for every input "state" for the local network
        
        """
        feed_dict={self.actor_critic.state: state}
                  
        q_value =self.tf_session.run([self.local_agent.value_output], feed_dict=feed_dict)          
        
        return q_value
        
    
    
    def update_global_network(self,experience_buffer):

        """        
        To update the global agent we first need to evaluate/train local agent to obtain
        
        1. policy (Theta') and value loss(Theta_value'), entropy and total loss
        2. then obtain local gradients and variables
        3. Compute the gradients with the loss function 
        3. assign local grads_and_variable to global grads and variables
        
        For this purpose we have set up necessary tensorflow "ops" in the actor_critic network with local agents "scope"
        
        Here we simply evaluate these network "ops" using local (i.e. agent) scope to update the global agent
        """
        experience_buffer=np.array(experience_buffer)

        # need to bootstrap from the last value if not terminal
        terminal=experience_buffer[-1][4]  
        
        if terminal: 
            R=0
        
        if not terminal:
            
            R=self.predict_value(experience_buffer[-1][0])  #last state for bootstrapping
                
        states=[]
        actions=[]
        target_v=[]
        advantages=[]
        
        for experience in (experience_buffer[:,0:-1]):
            
            state=experience[0]
            action=experience[1]
            reward=experience[2]
            
            R=reward+self.gamma*R
            policy_target=reward-self.predict_value(np.reshape(state,[1,self.state_size]))                        
            
            states.append(np.reshape(state,[1,self.state_size]))
            actions.append(action)
            target_v.append(R)
            advantages.append(policy_target)
        
        
            feed_dict={
                       self.local_agent.state: states,
                       self.local_agent.action: actions,
                       self.local_agent.target: target_v,
                       self.local_agent.advantage: advantages
                      }
                
                  
        # update the global network by evaluting the policy and value losses, 
        # creating acculmating the local agent's gradeints
        # Then applying local agents to global network         
        # This has been accounted for in the code for Actor Critic Network, (using scope) and thus we 
        # we can feed network with the necessary tensor "ops" and the values        

        value_loss,policy_loss, entropy,gradient_norm, _ = self.session.run([
                             self.local_agent.value_loss,
                             self.local_agent.policy_loss,
                             self.local_agent.entropy,
                             self.local_agent.grad_norms,
                             self.local_agent.apply_grads],
                             feed_dict=feed_dict)
        
            
        # can add these to summary network if required    
        
        return None     
        
        
    def swap_tf_ops(from_scope,to_scope):
            
             from_vars=tf.get_collection(tf.GraphKey.TRAINABLE_VARIABLES, from_scope)
             to_vars=tf.get_collection(tf.GraphKey.TRAINABLE_VARIABLES, to_scope)
             
             
             swaped=[]
             
             for from_var, to_var in zip(from_vars,to_vars):
                 swaped.append(to_var.assign(from_var))
                 
                 
             return swaped 