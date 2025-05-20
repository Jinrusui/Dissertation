from google.generativeai.types import HarmCategory, HarmBlockThreshold 
import google.generativeai as genai 
from openai import OpenAI 
import anthropic 
import base64 
import json 
import cv2 
import re 
import os 
import logging
from typing import Optional, Dict, Any, List, Tuple
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import numpy as np
import torch
# Set up logging
logger = logging.getLogger("atari_gpt.llms")

class Agent(): 
    def __init__(self, model_name=None, model=None, system_message=None, env=None): 
        """
        Initialize the Agent with the specified model and environment.
        
        Args:
            model_name: The specific model name to use
            model: The model provider key ('gpt4', 'gpt4o', 'claude', 'gemini')
            system_message: The system prompt to use
            env: The Gymnasium environment
        """
        self.model_key = model 
        logger.info(f'Model Key: {self.model_key}') 

        self.model_name = model_name 
        logger.info(f'Model Name: {self.model_name}') 

        self.messages = [] 
        self.system_message = system_message 
        self.env = env 
        self.action_space = self.env.action_space.n 
        self.reset_count = 0 

        # Initialize the appropriate client based on the model key
        try:
            if self.model_key in ['gpt4o', 'gpt4']: 
                self._init_openai_client()
            elif self.model_key == 'claude':
                self._init_anthropic_client()
            elif self.model_key == 'gemini':
                self._init_gemini_client()
#########support for qwen model#########################
            elif self.model_key == 'qwen':
                self._init_qwen_client()
            else:
                raise ValueError(f"Unsupported model key: {self.model_key}")
        except Exception as e:
            logger.error(f"Error initializing {self.model_key} client: {str(e)}")
            raise

    def _init_openai_client(self):
        """Initialize the OpenAI client with API key."""
        try:
            api_key = self._get_api_key("OPENAI_API_KEY.txt")
            self.client = OpenAI(api_key=api_key) 

            if self.system_message is not None: 
                system_prompt = {"role": "system", "content": [self.system_message]} 
                self.messages.append(system_prompt)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise

    def _init_anthropic_client(self):
        """Initialize the Anthropic client with API key."""
        try:
            api_key = self._get_api_key("ANTHROPIC_API_KEY.txt")
            self.client = anthropic.Anthropic(api_key=api_key)
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {str(e)}")
            raise

    def _init_gemini_client(self):
        """Initialize the Google Gemini client with API key."""
        try:
            api_key = self._get_api_key("GOOGLE_API_KEY.txt")
            genai.configure(api_key=api_key)
            generation_config = genai.GenerationConfig(temperature=1)
            
            if self.system_message is not None:
                self.client = genai.GenerativeModel(
                    model_name=self.model_name, 
                    system_instruction=self.system_message, 
                    generation_config=generation_config
                )
            else:
                self.client = genai.GenerativeModel(
                    model_name=self.model_name, 
                    generation_config=generation_config
                )
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {str(e)}")
            raise

#########support for qwen model#########################
    def _init_qwen_client(self):
        """Initialize the local Qwen model using Qwen2_5_VLForConditionalGeneration."""
        try:
            # model_path = self.model_name if self.model_name else "Qwen/Qwen2.5-VL-3B-Instruct"
            model_path = "Qwen/Qwen2.5-VL-3B-Instruct" # Directly use the 3B model
            logger.info(f"Loading Qwen model and processor from {model_path}")

            # Load the specific processor and model for Qwen2.5-VL
            self.qwen_processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16, # Use float16 for efficiency
                device_map="auto",        # Automatically distribute model across available devices
                trust_remote_code=True,
                attn_implementation="sdpa" # Use SDPA for potential speedup if available [cite: 7]
            )

            # Store system message if provided (though Qwen uses it differently in chat)
            if self.system_message is not None:
                self.qwen_system_message = self.system_message
            else:
                # Default system message (might not be directly used in the same way)
                self.qwen_system_message = "You are a game playing assistant and will be provided an image. This will be of the game Breakout, your goal is to provide me with what you believe to be the best action I could take to beat the game. Think about all possible actions and why each action is or is not the best action to take. The potential actions I can take are ‘0’: NOOP ‘1’: FIRE ‘2’: RIGHT ‘3’: LEFT. Provide output as a json structured as {reasoning: reasoning for actions and why to choose an action, action: The environment action which would provide the best next state}. The action key should only have the action I should take for the current frame as a number."

            # Reset the standard message list for the new model type
            self.messages = []
            # Add system message if using a structure that supports it (optional for Qwen API)
            # If the base prompt for the game already contains instructions, a separate system message might be less critical.

        except Exception as e:
            logger.error(f"Failed to initialize Qwen client: {str(e)}")
            raise

    def _get_api_key(self, key_file: str) -> str:
        """
        Get API key from file or environment variable.
        
        Args:
            key_file: The file containing the API key
            
        Returns:
            The API key as a string
        
        Raises:
            FileNotFoundError: If the key file doesn't exist and no environment variable is set
        """
        # Try to get from environment variable first
        env_var = key_file.replace(".txt", "")
        api_key = os.environ.get(env_var)
        
        if api_key:
            return api_key
            
        # Fall back to file
        try:
            with open(key_file, "r") as file:
                return file.read().strip()
        except FileNotFoundError:
            logger.error(f"API key file {key_file} not found and {env_var} environment variable not set")
            raise FileNotFoundError(f"API key file {key_file} not found and {env_var} environment variable not set")

    def encode_image(self, cv_image):
        _, buffer = cv2.imencode(".jpg", cv_image)
        return base64.b64encode(buffer).decode("utf-8")
    
    def query_LLM(self):
        print("\nInput History for LLM Query:")
        for message in self.messages:
            print(message)
        # Check which model to use and prompt the model 
        if self.model_key=='gpt4' or self.model_key=='gpt4o':
            self.response = self.client.chat.completions.create(
                model=self.model_name,
                response_format={ "type": "json_object" },
                messages=self.messages,
                temperature=1,
            )

        elif self.model_key == 'claude':
            if self.system_message is not None:
                self.response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=4096,
                    temperature=1,
                    system=self.system_message,
                    messages=self.messages,
                )
            else:
                self.response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=4096,
                    temperature=1,
                    messages=self.messages,
                )

        elif self.model_key == 'gemini':
            self.response = self.client.generate_content(self.messages,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            })
#########support for qwen model#########################
        elif self.model_key == 'qwen':
            try:
                # Prepare messages for the template - Extract text/image info
                # Note: Qwen processor handles the image linking during processing
                # We just need the structured message list.

                # Use apply_chat_template to format the prompt correctly
                # It expects the self.messages list directly
                text_prompt = self.qwen_processor.apply_chat_template(
                    self.messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                # Extract the last user image to pass to the processor
                # Assumes the structure in run_experiments sends one image per relevant turn
                image_to_process = None
                if hasattr(self, 'last_user_image'):
                    image_to_process = self.last_user_image

                if image_to_process is None:
                    # Find the last image in the message history if last_user_image wasn't set
                    for msg in reversed(self.messages):
                        if msg['role'] == 'user':
                            for item in msg['content']:
                                if item['type'] == 'image' and 'image_obj' in item:
                                    image_to_process = item['image_obj']
                                    break
                        if image_to_process:
                            break

                if image_to_process is None:
                    logger.error("No image found to process for Qwen VL model")
                    # Handle error: maybe return default or raise exception
                    # For atari_gpt, an image should always be present in user turns needing action.
                    raise ValueError("Qwen VL model requires an image input for game turns.")


                # Process text and image(s) using the processor instance
                inputs = self.qwen_processor(
                    text=[text_prompt], # Pass prompt as a list
                    images=[image_to_process], # Pass image as a list
                    return_tensors="pt"
                ).to(self.qwen_model.device) # Move inputs to the model's device

                # Generate response
                generated_ids = self.qwen_model.generate(
                    **inputs,
                    max_new_tokens=4096 # Or another appropriate value [cite: 9]
                    # Add other generation parameters if needed (e.g., temperature)
                )

                # Trim input tokens from generated IDs
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
                ]


                # Decode the response
                decoded_text = self.qwen_processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False # As per docs example [cite: 9]
                )[0] # Get the first response string

                # Store response in the expected format for clean_response
                self.response = {"text": decoded_text}
                print('\n\nresponse: ', self.response)

            except Exception as e:
                logger.error(f"Error querying Qwen model: {str(e)}")
                # Set response to None or empty to indicate failure
                self.response = {"text": ""} # Or handle error more robustly
                raise # Re-raise the exception for the main loop to handle if needed

        else:
            print('Incorrect Model name given please give correct model name')

        self.reset_count = 0

        # return the output of the model
        return self.response
    
    def reset_model(self):

        self.client = None

        if self.reset_count >= 3:
            return
        
        if self.model_key == 'gpt4o' or self.model_key == 'gpt4':
            file = open("OPENAI_API_KEY.txt", "r")
            api_key = file.read()
            self.client = OpenAI(api_key=api_key)

            if self.system_message is not None:
                system_prompt = {"role": "system", "content": [self.system_message]}
                self.messages.append(system_prompt)

        elif self.model_key == 'claude':
            file = open("ANTHROPIC_API_KEY.txt", "r")
            api_key = file.read()
            self.client = anthropic.Anthropic(api_key=api_key)
        
        elif self.model_key == 'gemini':
            file = open("GOOGLE_API_KEY.txt", "r")
            api_key = file.read()
            genai.configure(api_key=api_key)
            generation_config = genai.GenerationConfig(temperature=1)
            if self.system_message is not None:
                self.client = genai.GenerativeModel(model_name = self.model_name, system_instruction=self.system_message, generation_config=generation_config)
            else:
                self.client = genai.GenerativeModel(model_name = self.model_name, generation_config=generation_config)
#########support for qwen model#########################
        elif self.model_key == 'qwen':
            try:
                # For local models, we might just need to reset the history
                self.qwen_history = []
                logger.info("Reset Qwen chat history")
                
                # If model is acting strangely, we might want to reload it
                if self.reset_count >= 2:
                    # Reload model
                    logger.info("Reloading Qwen model")
                    model_path = self.model_name if self.model_name else "Qwen/Qwen2.5-VL"
                    self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                    self.client = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        device_map="auto",
                        trust_remote_code=True,
                        torch_dtype=torch.float16
                    )
            except Exception as e:
                logger.error(f"Error resetting Qwen model: {str(e)}")

        self.reset_count += 1

        print('Model is re-initiated...')

    def clean_model_output(self, output):
        """
        Clean the model output to ensure it's valid JSON.
        
        Args:
            output: The raw output from the model
            
        Returns:
            Cleaned output string
        """
        if not output:
            logger.warning("Received empty output from model")
            return ""
            
        # Remove any unescaped newline characters within the JSON string values
        cleaned_output = re.sub(r'(?<!\\)\n', ' ', output)
        
        # Replace curly quotes with straight quotes if necessary
        cleaned_output = cleaned_output.replace('"', '"').replace('"', '"')
        
        return cleaned_output

    def clean_response(self, response, path):
        """
        Extract and clean the response from the model.
        
        Args:
            response: The raw response object from the model
            path: Path to save the response for debugging
            
        Returns:
            Parsed JSON response or None if parsing fails
        """
        try:
            # Correctly get the response from model
            if self.model_key in ['gpt4', 'gpt4o']:
                response_text = response.choices[0].message.content
            elif self.model_key == 'claude':
                response_text = response.content[0].text
            elif self.model_key == 'gemini':
                response_text = response.text
            elif self.model_key == 'qwen':
                response_text = response["text"]
            else:
                raise ValueError(f"Unknown model key: {self.model_key}")
            
            if response_text is None:
                logger.warning("Received None response, attempting to get response again")
                response_text = self.get_response()

            response_text = self.clean_model_output(response_text)

            # Save the raw response for debugging
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path+'all_responses.txt', "a") as file:
                file.write(str(response_text) + '\n\n')

            # This regular expression finds the first { to the last }
            pattern = r'\{.*\}'
            # Search for the pattern
            match = re.search(pattern, response_text, flags=re.DOTALL)
            # Return the matched group which should be a valid JSON string
            if match:
                response_text = match.group(0)

            try:
                return json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {str(e)}")
                logger.info("Reprompting the model for valid JSON")

                # Create error message to reprompt the model
                error_message = 'Your output should be in a valid JSON format with a comma after every key-value pair except the last one. Please provide a response with the format: {"reasoning": "your reasoning here", "action": numeric_action_value}'
                
                # Add the error message to the context
                self.add_user_message(user_msg=error_message)

                logger.info('Generating new response...')
                retry_count = 0
                max_retries = 3
                
                while retry_count < max_retries:
                    try:
                        response = self.query_LLM()
                        logger.info('Proper response was generated')
                        return self.clean_response(response, path)  # Recursively process the new response
                    except Exception as e:
                        logger.error(f"Error generating response (attempt {retry_count+1}): {str(e)}")
                        self.reset_model()
                        retry_count += 1
                
                logger.error("Failed to get valid JSON after multiple attempts")
                return None
                
        except Exception as e:
            logger.error(f"Error cleaning response: {str(e)}")
            return None
    
    def check_action(self, response_text):
        """
        Validate the action from the model response.
        
        Args:
            response_text: The parsed JSON response from the model
            
        Returns:
            Valid action integer or None if invalid
        """
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            # Check if the response is a dictionary
            if isinstance(response_text, dict):
                # Check if the key action exists 
                if "action" in response_text:
                    try:
                        action = int(response_text["action"])
                        # Check if the action is valid for the environment
                        if 0 <= action < self.env.action_space.n:
                            return action
                        else:
                            logger.warning(f"Invalid action value: {action}. Must be between 0 and {self.env.action_space.n-1}")
                    except (ValueError, TypeError) as e:
                        logger.error(f"Error converting action to integer: {str(e)}")
                else:
                    logger.warning("Response missing 'action' key")
            else:
                logger.warning(f"Response is not a dictionary: {type(response_text)}")
            
            # If we get here, the action was invalid
            error_message = f'Your action value is invalid. Please provide a valid action between 0 and {self.env.action_space.n-1}.'
            self.add_user_message(user_msg=error_message)
            
            try:
                response = self.query_LLM()
                response_text = self.clean_response(response, "./")
                retry_count += 1
            except Exception as e:
                logger.error(f"Error getting new response: {str(e)}")
                retry_count += 1
        
        # If we've exhausted retries, return a default action (0)
        logger.error("Failed to get valid action after multiple attempts, using default action 0")
        return 0

    def get_response(self):
        # Check to see if you get a response from the model
        try: 
            response = self.query_LLM()


        # If there is an error with generating a response (internal error)
        # Reset the model and try again
        except:
            print('\n\nReceived Error when generating response reseting model\n\n')
            
            # Reset model
            self.reset_model()

            while True:

                # See if you get the correct output
                try:
                    response = self.query_LLM()
                    print('\n\nReceived correct output continuing experiment.')
                    break
                # If it doesn't then reset the model
                except:
                    # Create error message to reprompt the model
                    error_message = 'Please provide a proper output'
                    
                    # Add the error message to the context
                    self.add_user_message(user_msg=error_message)

                    print('Re-initiating model...')
                    self.reset_model() 

                    # This means that more than likely you ran out of credits so break the code to not spend money
                    if self.reset_count >= 3:
                        return None
                    
        if response == 'idchoicescreatedmodelobjectsystem_fingerprintusage':
            response = self.get_response()
        
        return response


    def generate_response(self, path) -> str:   
        response = self.get_response()

        # Check if it is just reasoning or actual action output
        self.path = path

        response_text = self.clean_response(response, path)
        print('\n\nresponse: ', response_text)

        action_output = self.check_action(response_text)

        return action_output, response_text

    def add_user_message(self, frame=None, user_msg=None):
        if self.model_key == 'gpt4' or self.model_key == 'gpt4o':
            if user_msg is not None and frame is not None:
                self.messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_msg},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{self.encode_image(frame)}",
                                    "detail": "low",
                                },
                            },
                        ],
                    }
                )
            elif user_msg is not None and frame is None:
                self.messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_msg},
                        ],
                    }
                )
            elif user_msg is None and frame is not None:
                self.messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{self.encode_image(frame)}",
                                    "detail": "low",
                                },
                            },
                        ],
                    }
                )
            else:
                pass
        
        elif self.model_key == 'claude':
            if frame is not None and user_msg is not None:
                image_data = self.encode_image(frame)
                self.messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_data
                                }
                            },
                            {
                                "type": "text",
                                "text": user_msg
                            }
                        ]
                    }
                )
            elif frame is not None:
                image_data = self.encode_image(frame)
                self.messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_data
                                }
                            }
                        ]
                    }
                )
            elif user_msg is not None:
                self.messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_msg
                            }
                        ]
                    }
                )

        if self.model_key == 'gemini':
            if frame is not None and user_msg is not None:
                image_data = self.encode_image(frame)
                self.messages.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "mime_type": "image/jpeg",
                                "data": image_data
                            },
                            {
                                "text": user_msg
                            }
                        ]
                    }
                )
            elif frame is not None and user_msg is None:
                image_data = self.encode_image(frame)
                self.messages.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "mime_type": "image/jpeg",
                                "data": image_data
                            }
                        ]
                    }
                )
            elif frame is None and user_msg is not None:
                self.messages.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": user_msg
                            }
                        ]
                    }
                )
            else:
                pass
#####################support for qwen model#########################
        elif self.model_key == 'qwen':
            message_content = []
            current_image = None # Keep track of the image for this message

            if frame is not None:
                # Convert frame (numpy array BGR) to PIL Image (RGB)
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                current_image = pil_image # Store the image object

            # IMPORTANT: Qwen expects image *before* text in the content list for a turn
            if current_image is not None:
                # Add a placeholder for the image object, will be handled during query_LLM
                message_content.append({"type": "image", "image_obj": current_image})

            if user_msg is not None:
                message_content.append({"type": "text", "text": user_msg})

            if message_content:
                self.messages.append({"role": "user", "content": message_content})
                # Store the image associated with the last user message for query_LLM
                self.last_user_image = current_image
            else:
                self.last_user_image = None # No image in this user turn
    def add_assistant_message(self, demo_str=None):

        if self.model_key =='gpt4' or self.model_key =='gpt4o':
            if demo_str is not None:
                self.messages.append({"role": "assistant", "content": self.response})
                demo_str = None
                return
            
            if self.response is not None:
                self.messages.append({"role": "assistant", "content": self.response})
        
        elif self.model_key == 'claude':
            if demo_str is not None:
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": demo_str},
                        ]
                    }
                )
                demo_str = None
                return

            if self.response is not None:
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": self.response.content[0].text},
                        ]
                    }
                )

        elif self.model_key =='gemini':
            if demo_str is not None:
                self.messages.append(
                    {
                        "role": "model",
                        "parts": demo_str
                    }
                )
                demo_str = None
                return

            if self.response is not None:
                assistant_msg = self.response.text
                self.messages.append(
                    {
                        "role": "model",
                        "parts": assistant_msg
                    }
                )
        #######support for qwen model#########################
        elif self.model_key == 'qwen':
                if demo_str is not None: # Handling demo string if needed
                    self.messages.append({"role": "assistant", "content": [{"type": "text", "text": demo_str}]})
                    return

                # Assumes self.response is updated in query_LLM to hold {"text": response_text}
                if self.response and isinstance(self.response, dict) and "text" in self.response:
                    response_text = self.response["text"]
                    # The clean_response function should ensure response_text is the JSON part
                    self.messages.append({"role": "assistant", "content": [{"type": "text", "text": response_text}]})

        else:
            self.messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": ' '},
                    ]
                }
            )


    def delete_messages(self):
        print('Deleting Set of Messages...')

        if self.model_key == 'gpt4' or self.model_key == 'gpt4o':
            message_len = 9
        else:
            message_len = 8

        
        if len(self.messages) >= message_len:

            if self.messages[0]['role'] == 'system':
                # Delete user message
                value = self.messages.pop(1)

                # Delete Assistant message
                value = self.messages.pop(1)

            else:
                # Delete user message
                self.messages.pop(0)

                # Delete Assistant message
                self.messages.pop(0)