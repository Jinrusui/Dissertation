import gymnasium as gym
import time
import json
from tqdm import tqdm

from local_llms import Agent
import cv2
import csv
import os
import sys
import pickle
from gymnasium.wrappers import RecordVideo, OrderEnforcing
import numpy as np

# Custom wrapper to prevent video closing on reset
class ContinuousRecordVideo(RecordVideo):
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        self.video_recorder.capture_frame()
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        if self.video_recorder:
            self.video_recorder.capture_frame()
        return observation, info

class run():
    def __init__(self, env_name, prompt, model, output_dir="./experiments/"):
      self.model_name = model
      self.rewards = 0
      self.cum_rewards = []
      self.action_list = []
      self.header = ["actions", "cumulative_rewards"]
      self.MODELS = {"OpenAI": ["gpt-4-turbo", "gpt-4o-2024-11-20"], 
              "Anthropic": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307", "claude-3.5-sonnet", "max-tokens-3-5-sonnet-2024-07-15"], 
              "Google": ["gemini-1.5-pro-latest", "gemini-pro", "gemini-pro-vision", "gemini-1.5-flash-latest"], 
              "Meta": ["llama3-70b-8192", "llama3-8b-8192"]
              }
      
      self.states = []
      
      self.steps_taken = 0

      # System prompt 
      self.sys_prompt = prompt

      self.env_name = env_name

      # Get rid of ALE/ for creating folder
      if "ALE/" in env_name:
          self.temp_env_name = env_name[4:]
      else:
          self.temp_env_name = env_name

      if self.temp_env_name == 'Frogger':
          self.pause = 130
          self.buffer_pause = 134
      else:
          self.pause = 15
          self.buffer_pause = 19

      # Total number of timesteps
      self.num_timesteps = 1000

      # Create new experiment folders path with model name 
      self.output_dir = output_dir
      self.new_dir = os.path.join(self.output_dir, self.temp_env_name[:-3] + '_'+ model +'/')

      # Create folders if they do not exist
      os.makedirs(os.path.dirname(self.new_dir), exist_ok=True)

      # Check if the environment state is saved
      if os.path.exists(self.new_dir + 'env_' + self.temp_env_name[:-3]+ '_state.pkl'):
          
          print('\n\nEnvironment Results Already Exist, Going to Next Environment...\n\n')
          return

      # Create Environment
      temp_env = gym.make(env_name, render_mode="rgb_array")

      # Apply the OrderEnforcer wrapper
      temp_env = OrderEnforcing(temp_env, disable_render_order_enforcing=True)

      # Reset the environment before any rendering
      temp_env.reset()

      # Record video
      self.env = ContinuousRecordVideo(env=temp_env, video_folder=self.new_dir, name_prefix=self.temp_env_name[:-3]+"_rollout")

      if self.model_name == 'rand':
          self.rand_rollout()

      elif self.model_name == 'gpt4':
          self.model = Agent(model_name=self.MODELS["OpenAI"][0], model = self.model_name, system_message=self.sys_prompt, env=self.env)
         
      elif self.model_name == 'gpt4o':
          self.model = Agent(model_name=self.MODELS["OpenAI"][1], model = self.model_name, system_message=self.sys_prompt, env=self.env)
            
      elif self.model_name == 'gemini':
          self.model = Agent(model_name=self.MODELS["Google"][3], model = self.model_name, system_message=self.sys_prompt, env=self.env)
            
      elif self.model_name == 'claude':
          self.model = Agent(model_name=self.MODELS["Anthropic"][2], model = self.model_name, system_message=self.sys_prompt, env=self.env)
      elif self.model_name == 'qwen':
            self.model = Agent(model_name="Qwen/Qwen2.5-VL-3B-Instruct", model = self.model_name, system_message=self.sys_prompt, env=self.env)
      if self.model_name != 'rand':   
          self.model_rollout()

      with open(self.new_dir + 'actions_rewards.csv', 'w') as f:
          writer = csv.writer(f)
          writer.writerow(self.header)
          
          for action, cum_reward in zip(self.action_list, self.cum_rewards):
              writer.writerow([action, cum_reward])

    def save_states(self, rewards, action):

        # Save the environment's 
        state = self.env.ale.cloneState()

        # Save the environment's random state
        random_state = self.env.np_random if hasattr(self.env, 'np_random') else self.env.unwrapped.np_random

        self.states.append((state, random_state, rewards, self.steps_taken, action))
        
        # Save the state to pkl file 
        with open(self.new_dir + 'env_' + self.temp_env_name[:-3]+ '_state.pkl', 'wb') as f:
            pickle.dump(self.states, f)

    def rand_rollout(self):
        # Start the recorder
        self.env.start_video_recorder()

        observation, info = self.env.reset()
        
        # Save the initial state
        self.save_states(self.rewards, 0)
        progress_bar = tqdm(total=self.num_timesteps, desc=f"Random Rollout ({self.temp_env_name})", unit="steps")
        for n in range(self.num_timesteps-self.steps_taken):
            observation = cv2.resize(observation, (512, 512))

            if n < self.pause:
                action = 0
                self.action_list.append(action)
                observation, reward, terminated, truncated, info = self.env.step(action)
                # Save the state once the action has been performed
                self.save_states(self.rewards, action)

                self.rewards += reward
                self.cum_rewards.append(self.rewards)
            
            elif n % 2 == 1:
                # image buffer
                action = self.env.action_space.sample()
                self.action_list.append(action)

                observation, reward, terminated, truncated, info = self.env.step(action)
                # Save the state once the action has been performed
                self.save_states(self.rewards, action)

                self.rewards += reward
                self.cum_rewards.append(self.rewards)
                
                if terminated or truncated:
                    observation, info = self.env.reset()
            else:
                action = 0
                self.action_list.append(action)
                observation, reward, terminated, truncated, info = self.env.step(action)
                self.env.render()

                # Save the state once the action has been performed
                self.save_states(self.rewards, action)

                self.rewards += reward
                self.cum_rewards.append(self.rewards)

                if terminated or truncated:
                        observation, info = self.env.reset() 

            self.steps_taken += 1
            progress_bar.update(1)
            progress_bar.set_postfix({"reward": self.rewards})
        
        # Close progress bar
        progress_bar.close()
        
        print('The reward for ' + self.env_name + ' is: ' + str(self.rewards))
        
        # Close the environment recorder
        self.env.close_video_recorder()
        
        # Close the environment
        self.env.close()

    def model_rollout(self):
        usr_msg1 = 'Analyze this game frame and select the optimal action. Focus on immediate gameplay elements visible in this specific frame, and follow the format: {"reasoning": "detailed step-by-step analysis", "action": X}'
        
        # Start the recorder
        self.env.start_video_recorder()

        observation, info = self.env.reset()
        
        # Save the initial state
        self.save_states(self.rewards, 0)
        progress_bar = tqdm(total=self.num_timesteps, desc=f"{self.model_name} Rollout ({self.temp_env_name})", unit="steps")


        for n in range(self.num_timesteps-self.steps_taken):
            
            # resize cv2 512x512
            observation = cv2.resize(observation, (512, 512))

            if n < self.pause:
                # Perform no-op action
                action = 0
                
                # Save action 
                self.action_list.append(action)

                # Perform Action
                observation, reward, terminated, truncated, info = self.env.step(action)

                self.env.render()

                # Sum reward and save
                self.rewards += reward
                self.cum_rewards.append(self.rewards)

                # Check done condition
                if terminated or truncated:
                        observation, info = self.env.reset()

            elif n % 2 == 1:

                # Create buffer of 4 frames
                if n < self.buffer_pause:

                    # Add frame and reason
                    self.model.add_user_message(observation, usr_msg1)

                    # Get response from model with action
                    action, full_response = self.model.generate_response(self.new_dir)

                    # Add models reasoning to context
                    self.model.add_assistant_message()
                    
                    # Save action
                    self.action_list.append(action)

                    # Perform Action
                    observation, reward, terminated, truncated, info = self.env.step(action)

                    self.env.render()
                    
                    # Sum reward and save
                    self.rewards += reward
                    self.cum_rewards.append(self.rewards)

                    # Check done condition 
                    if terminated or truncated:
                        observation, info = self.env.reset()
                
                else:
                    # Add frame and reason
                    self.model.add_user_message(observation, usr_msg1)

                    # Have model reason from the given image
                    action, full_response = self.model.generate_response(self.new_dir)

                    # Add models reasoning to context
                    self.model.add_assistant_message()
                
                    # Save action
                    self.action_list.append(action)
                    
                    # Perform Action
                    observation, reward, terminated, truncated, info = self.env.step(action)

                    self.env.render()

                    # Sum reward and save
                    self.rewards += reward
                    self.cum_rewards.append(self.rewards)

                    # Context buffer of only the 4 most recent frames
                    # delete oldest context
                    self.model.delete_messages()
                    
                    # Check done condition 
                    if terminated or truncated:
                        observation, info = self.env.reset()
            
            else:
                # Perform no-op action
                action = 0
                
                # Save action 
                self.action_list.append(action)

                # Perform Action
                observation, reward, terminated, truncated, info = self.env.step(action)

                self.env.render()

                # Sum reward and save
                self.rewards += reward
                self.cum_rewards.append(self.rewards)

                # Check done condition
                if terminated or truncated:
                        observation, info = self.env.reset() 
            
            # Save the state once the action has been performed
            self.save_states(self.rewards, action)

            self.steps_taken += 1
            progress_bar.update(1)
            progress_bar.set_postfix({"reward": self.rewards})
        
        # Close progress bar
        progress_bar.close()
        
        # Close the environment recorder
        self.env.close_video_recorder()
        
        # Close the environment
        self.env.close()
