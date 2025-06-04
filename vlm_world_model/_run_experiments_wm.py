import os
import sys
import yaml
from pathlib import Path
import torch
from functools import partial

# Add parent directories to path to allow imports
project_dir = Path(__file__).parent.parent
parent_dir = project_dir.parent
sys.path.insert(0, str(project_dir))
sys.path.insert(0, str(parent_dir))

# Import from atari_gpt
from atari_gpt.run_experiments import run as BaseRun

# Import VLM+WM agent
from vlm_world_model.src.vlm_wm import VLMWorldModelAgent

# Import Tokenizer + WorldModel
from iris.src.models.tokenizer.tokenizer import Tokenizer
from iris.src.models.world_model import WorldModel

class RunWithWorldModel(BaseRun):
    """Extends the base Run class to use VLMWorldModelAgent with world model integration."""
    
    def __init__(self, env_name, prompt, model, output_dir="./experiments/", 
                 tokenizer=None, world_model=None, plan_horizon=5, num_plans=3, device="cuda"):
        """Initialize with additional world model parameters."""
        self.tokenizer = tokenizer
        self.world_model = world_model
        self.plan_horizon = plan_horizon
        self.num_plans = num_plans
        self.device = device

        self.Agent = partial(VLMWorldModelAgent, tokenizer=self.tokenizer, world_model=self.world_model,plan_horizon=self.plan_horizon, num_plans=self.num_plans, device=self.device)
        
        # Call parent __init__ with these parameters
        super().__init__(env_name, prompt, model, output_dir)
