import gym
from params import parse_args
from dqn import DQN
import numpy as np
from collections import deque
from fxEnviroment import fxEnviroment
from forex_DB import database
from datetime import datetime, timedelta





def run_dqn():
  # get command line arguments, defaults set in utils.py
  agent_params, dqn_params, cnn_params = parse_args()


  steps_to_update =  10           #agent_params['steps_to_update']
  
  current_time=datetime.utcnow()
  start_time=current_time-timedelta(days=10) 
  end_time=start_time-timedelta(days=7) 
      
  ticker_str=('AUD_CAD', 'AUD_CHF', 'AUD_HKD')
        
  SQL=database()
  env=fxEnviroment(SQL,start_time,end_time,ticker_str)

  observation_shape=env.current_state.shape
  num_actions=3


  # initialize dqn learning
  dqn = DQN(num_actions, observation_shape, dqn_params, cnn_params)

  episode_starttime=env.starttime  
  total_steps = 0
  while episode_starttime<env.endtime:
      
      episode_endtime=start_time+timedelta(minutes=60) 
      observation = env.reset(episode_starttime,episode_endtime)
      reward_sum = 0
      done=0


      while not done:
          
          # select action based on the model
          action = dqn.select_action(observation)
          # execute actin in emulator
          new_observation, reward, done, _ = env.step(action)
          # update the state 
          dqn.update_state(action, observation, new_observation, reward, done)
          observation = new_observation

          # train the model
          dqn.train_step()

          reward_sum += reward
          if done:
              print ("episode completed")
              print ("Reward for this episode: ", reward_sum)

              episode_starttime=episode_endtime
              break
             

          if total_steps % steps_to_update == 0:
            print ("updating target network...")
            dqn.update_target()

          
if __name__ == '__main__':
  run_dqn()
