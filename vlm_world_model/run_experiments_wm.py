import gymnasium as gym
import time
import json
from tqdm import tqdm

from vlm_world_model.src.vlm_wm import VLMWorldModelAgent
from vlm_world_model.src.action_buffer import ActionBuffer
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

class RunWithWorldModel():
    def __init__(self, env_name, prompt, model, output_dir="./experiments/", tokenizer=None, world_model=None, plan_horizon=5, num_plans=3, device="cuda", real_time_render=False, use_world_model=True, live_chat=False, num_timesteps=100, use_action_buffer=True, action_buffer_size=10, action_buffer_threshold=0.0):
      self.tokenizer = tokenizer
      self.world_model = world_model
      self.plan_horizon = plan_horizon
      self.num_plans = num_plans
      self.device = device
      self.real_time_render = real_time_render
      self.use_world_model = use_world_model
      self.live_chat = live_chat
      self.use_action_buffer = use_action_buffer
      self.action_buffer_size = action_buffer_size
      self.action_buffer_threshold = action_buffer_threshold
      

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
      self.num_timesteps = num_timesteps

      # Create new experiment folders path with model name 
      self.output_dir = output_dir
      self.new_dir = os.path.join(self.output_dir, self.temp_env_name[:-3] + '_'+ model +'/')

      # Create folders if they do not exist
      os.makedirs(os.path.dirname(self.new_dir), exist_ok=True)

      # Chat history for live chat window
      self.chat_history = []
      self.chat_window_height = 650
      self.chat_window_width = 500
      self.chat_window = np.ones((self.chat_window_height, self.chat_window_width, 3), dtype=np.uint8) * 240  # Light gray background
      self.font_size = 0.7  # Base font size
      self.title_font_size = 1.0  # Title font size
      self.font_thickness = 2  # Text thickness
      self.scroll_position = 0  # Current scroll position
      
      if self.real_time_render:
        cv2.namedWindow("Real-time Rendering", cv2.WINDOW_NORMAL)
        if self.live_chat:
            cv2.namedWindow("Live Chat", cv2.WINDOW_NORMAL)
            cv2.moveWindow("Real-time Rendering", 50, 50)
            cv2.resizeWindow("Real-time Rendering", 640, 640)  # Larger rendering window
            cv2.moveWindow("Live Chat", 50 + 640 + 20, 50)  # Position chat window to the right of render window
            cv2.resizeWindow("Live Chat", self.chat_window_width, self.chat_window_height)
            
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

      # Initialize action buffer if enabled
      self.action_buffer = ActionBuffer() if self.use_action_buffer else None

      if self.model_name == 'rand':
          self.rand_rollout()

      elif self.model_name == 'gpt4':
          self.model = VLMWorldModelAgent(
              model_name=self.MODELS["OpenAI"][0], 
              model=self.model_name, 
              system_message=self.sys_prompt, 
              env=self.env,
              tokenizer=self.tokenizer,
              world_model=self.world_model,
              plan_horizon=self.plan_horizon,
              num_plans=self.num_plans,
              device=self.device,
              use_world_model=self.use_world_model,
              action_buffer=self.action_buffer
          )
         
      elif self.model_name == 'gpt4o':
          self.model = VLMWorldModelAgent(
              model_name=self.MODELS["OpenAI"][1], 
              model=self.model_name, 
              system_message=self.sys_prompt, 
              env=self.env,
              tokenizer=self.tokenizer,
              world_model=self.world_model,
              plan_horizon=self.plan_horizon,
              num_plans=self.num_plans,
              device=self.device,
              use_world_model=self.use_world_model,
              action_buffer=self.action_buffer
          )
            
      elif self.model_name == 'gemini':
          self.model = VLMWorldModelAgent(
              model_name=self.MODELS["Google"][3], 
              model=self.model_name, 
              system_message=self.sys_prompt, 
              env=self.env,
              tokenizer=self.tokenizer,
              world_model=self.world_model,
              plan_horizon=self.plan_horizon,
              num_plans=self.num_plans,
              device=self.device,
              use_world_model=self.use_world_model,
              action_buffer=self.action_buffer
          )
            
      elif self.model_name == 'claude':
          self.model = VLMWorldModelAgent(
              model_name=self.MODELS["Anthropic"][2], 
              model=self.model_name, 
              system_message=self.sys_prompt, 
              env=self.env,
              tokenizer=self.tokenizer,
              world_model=self.world_model,
              plan_horizon=self.plan_horizon,
              num_plans=self.num_plans,
              device=self.device,
              use_world_model=self.use_world_model,
              action_buffer=self.action_buffer
          )
      elif self.model_name == 'qwen':
            self.model = VLMWorldModelAgent(
                model_name="Qwen/Qwen2.5-VL-3B-Instruct", 
                model=self.model_name, 
                system_message=self.sys_prompt, 
                env=self.env,
                tokenizer=self.tokenizer,
                world_model=self.world_model,
                plan_horizon=self.plan_horizon,
                num_plans=self.num_plans,
                device=self.device,
                use_world_model=self.use_world_model,
                action_buffer=self.action_buffer
            )
      if self.model_name != 'rand':   
          self.model_rollout()

      with open(self.new_dir + 'actions_rewards.csv', 'w') as f:
          writer = csv.writer(f)
          writer.writerow(self.header)
          
          for action, cum_reward in zip(self.action_list, self.cum_rewards):
              writer.writerow([action, cum_reward])

    def _truncate_message(self, message, max_length=300):
        """Truncate long messages for better display"""
        if len(message) <= max_length:
            return message
        
        # Truncate and add ellipsis
        return message[:max_length] + "..."
        
    def update_chat_window(self, message=None, is_user=False, is_action=False):
        """Update the chat window with new messages"""
        if not self.live_chat:
            return
            
        if message:
            # Truncate long messages for better display
            truncated_message = self._truncate_message(message)
            
            # Add message to chat history
            if is_user:
                self.chat_history.append(("User", truncated_message))
            elif is_action:
                self.chat_history.append(("Action", truncated_message))
            else:
                self.chat_history.append(("Model", truncated_message))
            
            # Limit chat history to last 30 messages
            if len(self.chat_history) > 30:
                self.chat_history = self.chat_history[-30:]
                
            # Auto-scroll to bottom when new message is added
            self.scroll_position = max(0, len(self.chat_history) - 10)
        
        # Redraw chat window
        self.chat_window = np.ones((self.chat_window_height, self.chat_window_width, 3), dtype=np.uint8) * 240
        
        # Add title
        cv2.putText(self.chat_window, f"Live Chat - {self.model_name}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, self.title_font_size, (0, 0, 0), self.font_thickness)
        
        # Draw separator line
        cv2.line(self.chat_window, (0, 40), (self.chat_window_width, 40), (200, 200, 200), self.font_thickness)
        
        # Add scroll instructions
        cv2.putText(self.chat_window, "Keys: Up/Down arrows to scroll", (self.chat_window_width - 300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        # Draw messages
        y_pos = 70
        
        # Get visible messages based on scroll position
        visible_messages = self.chat_history[self.scroll_position:self.scroll_position+10]
        
        # Draw scroll indicator
        if self.scroll_position > 0:
            cv2.putText(self.chat_window, "▲ More messages above", (self.chat_window_width // 2 - 100, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
                        
        if self.scroll_position + 10 < len(self.chat_history):
            cv2.putText(self.chat_window, "▼ More messages below", (self.chat_window_width // 2 - 100, self.chat_window_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        for sender, msg in visible_messages:
            # Set color based on sender type
            if sender == "User":
                color = (0, 0, 255)  # Red for user
                bg_color = (240, 220, 220)  # Light red background
            elif sender == "Model":
                color = (0, 128, 0)  # Green for model
                bg_color = (220, 240, 220)  # Light green background
            else:  # Action
                color = (255, 0, 0)  # Blue for action
                bg_color = (220, 220, 240)  # Light blue background
            
            # Add sender name with background highlight
            sender_text = f"{sender}:"
            text_size = cv2.getTextSize(sender_text, cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_thickness)[0]
            
            # Draw background for sender
            cv2.rectangle(self.chat_window, (5, y_pos - 20), (15 + text_size[0], y_pos + 5), bg_color, -1)
            
            # Draw sender name
            cv2.putText(self.chat_window, sender_text, (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, self.font_size, color, self.font_thickness)
            y_pos += 30
            
            # Add message text with word wrapping and improved formatting
            words = msg.split()
            line = ""
            line_width = 0
            max_width = self.chat_window_width - 40  # Left margin of 20px, right margin of 20px
            
            # Calculate average character width based on font size
            char_width = int(self.font_size * 20)
            
            for word in words:
                word_width = len(word) * char_width
                
                # If adding this word would exceed the width
                if line_width + word_width > max_width:
                    # Draw background for text line
                    text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_thickness)[0]
                    cv2.rectangle(self.chat_window, (15, y_pos - 20), (25 + text_size[0], y_pos + 5), (235, 235, 235), -1)
                    
                    # Draw text
                    cv2.putText(self.chat_window, line, (20, y_pos), 
                                cv2.FONT_HERSHEY_SIMPLEX, self.font_size, (0, 0, 0), self.font_thickness)
                    y_pos += 30
                    line = word
                    line_width = word_width
                else:
                    if line:
                        line += " " + word
                        line_width += word_width + char_width  # Add space width
                    else:
                        line = word
                        line_width = word_width
            
            # Add the last line
            if line:
                # Draw background for text line
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_thickness)[0]
                cv2.rectangle(self.chat_window, (15, y_pos - 20), (25 + text_size[0], y_pos + 5), (235, 235, 235), -1)
                
                # Draw text
                cv2.putText(self.chat_window, line, (20, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, self.font_size, (0, 0, 0), self.font_thickness)
            
            y_pos += 40  # Extra space between messages
        
        # Display the chat window if real-time rendering is enabled
        if self.real_time_render and self.live_chat:
            cv2.imshow("Live Chat", self.chat_window)
            cv2.waitKey(1)

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
                
                if self.real_time_render:
                    render_frame = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Real-time Rendering", render_frame)
                    self.update_chat_window(f"No-op action: {action}", is_action=True)
                    cv2.waitKey(1)
                # Save the state once the action has been performed
                self.save_states(self.rewards, action)

                self.rewards += reward
                self.cum_rewards.append(self.rewards)
            
            elif n % 2 == 1:
                # image buffer
                action = self.env.action_space.sample()
                self.action_list.append(action)

                observation, reward, terminated, truncated, info = self.env.step(action)
                if self.real_time_render:
                    render_frame = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Real-time Rendering", render_frame)
                    self.update_chat_window(f"Random action: {action}", is_action=True)
                    cv2.waitKey(1)
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
                if self.real_time_render:
                    render_frame = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Real-time Rendering", render_frame)
                    self.update_chat_window(f"No-op action: {action}", is_action=True)
                    cv2.waitKey(1)
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

        if self.real_time_render:
            cv2.destroyWindow("Real-time Rendering")
            if self.live_chat:
                cv2.destroyWindow("Live Chat")
        
        print('The reward for ' + self.env_name + ' is: ' + str(self.rewards))
        
        # Close the environment recorder
        self.env.close_video_recorder()
        
        # Close the environment
        self.env.close()

    def save_chat_logs(self):
        """Save chat logs to a file"""
        if not self.live_chat or not self.chat_history:
            return
            
        chat_log_path = os.path.join(self.new_dir, f"{self.temp_env_name[:-3]}_chat_log.txt")
        
        try:
            with open(chat_log_path, 'w') as f:
                f.write(f"Chat Log for {self.env_name} with {self.model_name}\n")
                f.write("="*50 + "\n\n")
                
                for sender, msg in self.chat_history:
                    f.write(f"{sender}: {msg}\n\n")
                    
            print(f"Chat logs saved to {chat_log_path}")
        except Exception as e:
            print(f"Error saving chat logs: {str(e)}")
            
    def model_rollout(self):
        #usr_msg1 = 'Analyze this game frame and select the optimal action. Focus on immediate gameplay elements visible in this specific frame, and follow the format: {"reasoning": "detailed step-by-step analysis", "action": X}'
        
        # Start the recorder
        self.env.start_video_recorder()

        observation, info = self.env.reset()
        
        # Save the initial state
        self.save_states(self.rewards, 0)
        progress_bar = tqdm(total=self.num_timesteps, desc=f"{self.model_name} Rollout ({self.temp_env_name})", unit="steps")

        # Initialize chat window if real-time rendering is enabled
        if self.real_time_render and self.live_chat:
            self.update_chat_window("Starting game session...", is_user=True)
            self.update_chat_window("KEYBOARD CONTROLS:\n- Up Arrow or 'w': Scroll up\n- Down Arrow or 's': Scroll down\n- 'r': Reset scroll to bottom", is_user=False)
            if self.use_action_buffer:
                self.update_chat_window("Action buffer enabled: Plans with high scores will be stored for future use", is_user=False)

        try:
            for n in range(self.num_timesteps-self.steps_taken):
                
                # resize cv2 512x512
                observation = cv2.resize(observation, (512, 512))

                # Check for keyboard input for scrolling chat
                if self.real_time_render and self.live_chat:
                    key = cv2.waitKey(1) & 0xFF
                    if key == 82 or key == ord('w'):  # Up arrow or 'w'
                        self.scroll_position = max(0, self.scroll_position - 1)
                        self.update_chat_window()
                    elif key == 84 or key == ord('s'):  # Down arrow or 's'
                        self.scroll_position = min(self.scroll_position + 1, max(0, len(self.chat_history) - 10))
                        self.update_chat_window()
                    elif key == ord('r'):  # 'r' to reset scroll
                        self.scroll_position = max(0, len(self.chat_history) - 10)
                        self.update_chat_window()

                if n < self.pause:
                    # Perform no-op action
                    action = 0
                    
                    # Save action 
                    self.action_list.append(action)

                    # Perform Action
                    observation, reward, terminated, truncated, info = self.env.step(action)
                    if self.real_time_render:
                        render_frame = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
                        cv2.imshow("Real-time Rendering", render_frame)
                        self.update_chat_window(f"Warm-up phase: No-op action {action}", is_action=True)
                        cv2.waitKey(1)

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
                        # Update chat window with user message if real-time rendering is enabled
                        # if self.real_time_render and self.live_chat:
                        #     self.update_chat_window(usr_msg1, is_user=True)

                        # Add frame and reason
                        #self.model.add_user_message(observation, usr_msg1)

                        # Get response from model with action
                        action, full_response = self.model.generate_response(self.new_dir)
                        
                        # Update chat window with model response if real-time rendering is enabled
                        if self.real_time_render and self.live_chat:
                            # Extract reasoning from the response if available
                            reasoning = "Analyzing game frame..."
                            from_buffer = full_response.get("from_buffer", False)
                            
                            if from_buffer:
                                reasoning = "Using pre-validated action from buffer"
                                # Display buffer status if available
                                if "buffer_status" in full_response:
                                    buffer_status = full_response["buffer_status"]
                                    buffer_info = f"Buffer: {buffer_status['size']} plans, {buffer_status['actions_remaining']} actions remaining"
                                    self.update_chat_window(buffer_info)
                            elif isinstance(full_response, dict) and "reasoning" in full_response:
                                reasoning = full_response["reasoning"]
                            elif isinstance(full_response, dict) and "plans" in full_response and len(full_response["plans"]) > 0:
                                reasoning = full_response["plans"][0].get("explanation", "Selected best action based on world model simulation")
                            
                            self.update_chat_window(reasoning)
                            self.update_chat_window(f"Action: {action}", is_action=True)

                        # Add models reasoning to context
                        self.model.add_assistant_message()
                        
                        # Save action
                        self.action_list.append(action)

                        # Perform Action
                        observation, reward, terminated, truncated, info = self.env.step(action)
                        if self.real_time_render:
                            render_frame = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
                            cv2.imshow("Real-time Rendering", render_frame)
                            cv2.waitKey(1)

                        self.env.render()
                        
                        # Sum reward and save
                        self.rewards += reward
                        self.cum_rewards.append(self.rewards)

                        # Check done condition 
                        if terminated or truncated:
                            observation, info = self.env.reset()
                    
                    else:
                        # Update chat window with user message if real-time rendering is enabled
                        # if self.real_time_render and self.live_chat:
                        #     self.update_chat_window("What action should I take now?", is_user=True)
                        
                        # Add frame and reason
                        #self.model.add_user_message(observation, usr_msg1)

                        # Have model reason from the given image
                        action, full_response = self.model.generate_response(self.new_dir)
                        
                        # Update chat window with model response if real-time rendering is enabled
                        if self.real_time_render and self.live_chat:
                            # Extract reasoning from the response if available
                            reasoning = "Analyzing game frame..."
                            from_buffer = full_response.get("from_buffer", False)
                            
                            if from_buffer:
                                reasoning = "Using pre-validated action from buffer"
                                # Display buffer status if available
                                if "buffer_status" in full_response:
                                    buffer_status = full_response["buffer_status"]
                                    buffer_info = f"Buffer: {buffer_status['size']} plans, {buffer_status['actions_remaining']} actions remaining"
                                    self.update_chat_window(buffer_info)
                            elif isinstance(full_response, dict) and "reasoning" in full_response:
                                reasoning = full_response["reasoning"]
                            elif isinstance(full_response, dict) and "plans" in full_response and len(full_response["plans"]) > 0:
                                reasoning = full_response["plans"][0].get("explanation", "Selected best action based on world model simulation")
                            
                            self.update_chat_window(reasoning)
                            self.update_chat_window(f"Action: {action}", is_action=True)

                        # Add models reasoning to context
                        self.model.add_assistant_message()

                        # Save action
                        self.action_list.append(action)
                        
                        # Perform Action
                        observation, reward, terminated, truncated, info = self.env.step(action)
                        if self.real_time_render:
                            render_frame = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
                            cv2.imshow("Real-time Rendering", render_frame)
                            cv2.waitKey(1)

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
                    if self.real_time_render:
                        render_frame = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
                        cv2.imshow("Real-time Rendering", render_frame)
                        self.update_chat_window(f"Processing... No-op action {action}", is_action=True)
                        cv2.waitKey(1)

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
        finally:
            # Close progress bar
            progress_bar.close()

            if self.real_time_render:
                cv2.destroyWindow("Real-time Rendering")
                if self.live_chat:
                    cv2.destroyWindow("Live Chat")
                    # Save chat logs
                    self.save_chat_logs()
            
            # Close the environment recorder
            self.env.close_video_recorder()
            
            # Close the environment
            self.env.close()