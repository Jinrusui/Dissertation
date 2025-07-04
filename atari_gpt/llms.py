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
            package_dir = os.path.dirname(__file__)

            api_key_file_path = os.path.join(package_dir, key_file)
            with open(api_key_file_path, "r") as file:
                return file.read().strip()
        except FileNotFoundError:
            logger.error(f"API key file {key_file} not found and {env_var} environment variable not set")
            raise FileNotFoundError(f"API key file {key_file} not found and {env_var} environment variable not set")

    def encode_image(self, cv_image):
        _, buffer = cv2.imencode(".jpg", cv_image)
        return base64.b64encode(buffer).decode("utf-8")
    
    def query_LLM(self):

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

        elif self.model_key == 'gemini':
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

    def add_assistant_message(self, demo_str=None):
        if self.model_key in ['gpt4', 'gpt4o']:
            if demo_str is not None:
                self.messages.append({"role": "assistant", "content": demo_str})
                return

            if self.response is not None:
                try:
                    content = self.response.choices[0].message.content
                    self.messages.append({"role": "assistant", "content": content})
                except (AttributeError, IndexError):
                    logger.warning("Could not extract content from OpenAI response, skipping assistant message.")

        elif self.model_key == 'claude':
            if demo_str is not None:
                self.messages.append({"role": "assistant", "content": [{"type": "text", "text": demo_str}]})
                return

            if self.response is not None:
                try:
                    content = self.response.content[0].text
                    self.messages.append({"role": "assistant", "content": [{"type": "text", "text": content}]})
                except (AttributeError, IndexError):
                    logger.warning("Could not extract content from Anthropic response, skipping assistant message.")

        elif self.model_key == 'gemini':
            if demo_str is not None:
                self.messages.append({"role": "model", "parts": [{"text": demo_str}]})
                return

            if self.response is not None:
                try:
                    # Gemini response object might not have .text if the prompt was blocked.
                    assistant_msg = self.response.text
                    self.messages.append({"role": "model", "parts": [{"text": assistant_msg}]})
                except AttributeError:
                    logger.warning("Could not extract text from Gemini response, skipping assistant message.")
        else:
            # Fallback for unknown models, though it might not be useful
            self.messages.append({"role": "assistant", "content": " "})

    def manage_context_window(self, max_turns: int = 3):
        """
        Manages the conversation history to prevent it from growing indefinitely.
        Keeps the system prompt and the most recent N conversation turns.

        :param max_turns: The number of recent user/assistant turns to keep.
        """
        # A turn consists of one user message and one assistant message.
        max_convo_messages = max_turns * 2

        # Ensure there is a system message to preserve
        if not self.messages or self.messages[0].get("role") != "system":
            # For models like claude/gemini that might not have an explicit system message in the list
            # We just trim the oldest messages.
            if len(self.messages) > max_convo_messages:
                self.messages = self.messages[-max_convo_messages:]
            return

        # If there is a system message, preserve it
        # Check if the total number of messages exceeds the limit (1 system prompt + N conversation messages)
        if len(self.messages) > 1 + max_convo_messages:
            # print(f"Managing context window: trimming to system prompt and last {max_turns} turns.")
            
            # Create the new list of messages
            new_messages = [self.messages[0]] + self.messages[-max_convo_messages:]
            
            self.messages = new_messages

