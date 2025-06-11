"""VLM‑WorldModel Agent
====================
A drop‑in replacement for the existing *Agent* class that:
1.  Lets an image‑VLM (GPT‑4o / Gemini / Claude‑Vision …) **propose multiple multi‑step action plans**.
2.  Evaluates every plan inside an **offline world‑model** (tokenizer + world_model) to obtain cumulative reward.
3.  Selects the plan with the best predicted reward and returns **its first action** for real‑environment execution.

It keeps **exactly** the same public interface as the old `Agent` used in *run_experiments.py*:
```python
agent = VLMWorldModelAgent(model_name=..., model="gpt4o", system_message=..., env=env,
                           tokenizer=tok, world_model=wm, device="cuda:0")
action, full_response = agent.generate_response(path)   # unchanged call‑site
– `full_response` is now the raw JSON produced by the VLM.

The new agent subclasses the original `Agent` and **only overrides the upper‑level call‑chain** so other methods in *run_experiments* stay intact.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import sys
sys.path.insert(0, '/mnt/e/Projects/Dissertation')

from atari_gpt.llms import Agent  # parent base class (already on import path)
from iris.src.models.tokenizer.tokenizer import Tokenizer, TokenizerEncoderOutput
from iris.src.models.world_model import WorldModel, WorldModelOutput
from PIL import Image
import cv2
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from vlm_world_model.src.action_buffer import ActionBuffer

logger = logging.getLogger("vlm_wm_agent")

__all__ = ["VLMWorldModelAgent"]


def _rgb_to_bchw(rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    """H × W × C uint8 → 1 × C × H × W float32[0,1]."""
    if isinstance(rgb, str):
        # If we somehow got a string, this is an error
        raise TypeError(f"Expected image data, got string: {rgb[:30]}...")
    
    # Handle PIL Image
    if hasattr(rgb, 'convert'):  # Check if it's a PIL Image
        rgb = np.array(rgb)
    
    # Fast path for most common case (HWC uint8)
    if len(rgb.shape) == 3 and rgb.shape[2] == 3 and rgb.dtype == np.uint8:
        # Avoid unnecessary copies by using torch.as_tensor instead of from_numpy
        tensor = torch.as_tensor(rgb, device='cpu').permute(2, 0, 1).float().mul_(1.0/255.0).unsqueeze(0)
        return tensor.to(device, non_blocking=True)
    
    # Handle other cases
    if hasattr(rgb, 'dtype') and rgb.dtype != np.uint8:
        # If we have float values, ensure they're in [0,1] and convert to uint8
        rgb = (rgb.clip(0, 1) * 255).astype(np.uint8)
    
    # Handle different channel arrangements
    if len(rgb.shape) == 3 and rgb.shape[2] == 3:
        # Standard HWC format
        tensor = torch.as_tensor(rgb, device='cpu').permute(2, 0, 1).float().mul_(1.0/255.0).unsqueeze(0)
    elif len(rgb.shape) == 3 and rgb.shape[0] == 3:
        # Already in CHW format
        tensor = torch.as_tensor(rgb, device='cpu').float().mul_(1.0/255.0).unsqueeze(0)
    else:
        raise ValueError(f"Unexpected image shape: {rgb.shape}")
    
    return tensor.to(device, non_blocking=True)


class VLMWorldModelAgent(Agent):
    """LLM + world‑model agent. 100 % compatible with the old *Agent* API used in run_experiments.py.
    
    Extra args (over *Agent*):
    -------------------------
    tokenizer : Tokenizer  – pre‑trained image → token mapper.
    world_model : WorldModel  – pre‑trained autoregressive world model.
    plan_horizon : int  – how many steps each plan should contain.
    num_plans : int  – how many alternative plans VLM must propose.
    device : str | torch.device  – device for world‑model inference.
    action_buffer : ActionBuffer | None – buffer to store validated plans
    """

    # ---------------------------------------------------------------------
    # Construction ---------------------------------------------------------
    # ---------------------------------------------------------------------

    def __init__(
        self,
        *,
        tokenizer: Tokenizer,
        world_model: WorldModel,
        plan_horizon: int = 5,
        num_plans: int = 3,
        device: str | torch.device = "cuda:0",
        use_world_model: bool = True,
        action_buffer: ActionBuffer | None = None,
        **agent_kwargs: Any,
    ) -> None:
        super().__init__(**agent_kwargs)  # initialise LLM client etc.

        self.device = torch.device(device)
        self.tokenizer = tokenizer.to(self.device).eval()
        self.world_model = world_model.to(self.device).eval()

        self.plan_horizon = plan_horizon
        self.num_plans = num_plans
        self.use_world_model = use_world_model
        self.use_action_buffer = action_buffer is not None
        self.action_buffer = action_buffer or ActionBuffer()
        self._num_obs_tokens: int | None = None  # cache after first encode
        print("VLMWorldModelAgent initialized!!!!!")
    # ---------------------------------------------------------------------
    # PUBLIC API (unchanged signature) ------------------------------------
    # ---------------------------------------------------------------------

    def generate_response(self, path: str | Path) -> Tuple[int, Dict[str, Any]]:  # type: ignore[override]
        """Overrides parent: returns *(selected_action, raw_VLM_JSON)*.

        * path is still required because run_experiments writes raw completions there.
        * We store VLM raw output exactly like parent; then we compute best‑plan action.
        """
        # First check if there's an action in the buffer (fastest path)
        buffered_action = self.action_buffer.get_action()
        if buffered_action is not None:
            # Create a minimal response to maintain API compatibility
            vlm_json = {
                "reasoning": "Action taken from pre-validated plan in buffer",
                "action": buffered_action,
                "from_buffer": True,
                "buffer_status": self.action_buffer.get_buffer_status()
            }
            return buffered_action, vlm_json
        
        # If buffer is empty, generate new plans
        # === 1. Ask VLM for plans ===================================================
        vlm_json = self._query_vlm_for_plans(path)
        plans = self.check_plans(vlm_json)
        if not plans:
            return 0, vlm_json  # Fallback to NOOP

        # === 2. Evaluate via world‑model ============================================
        if not self.use_world_model:
            # Fast path when world model is disabled
            best_action = int(plans[0]['actions'][0])
            vlm_json["buffer_status"] = self.action_buffer.get_buffer_status()
            return best_action, vlm_json
            
        # Get observation and evaluate plans
        obs_rgb = self.env.render()
        best_action, best_plan_idx, plan_scores = self._pick_best_action_and_plan(obs_rgb, plans)
        
        # === 3. Store the best plan in the buffer =================================
        if self.use_action_buffer and plan_scores:
            # Find best valid plan
            best_score = float('-inf')
            best_plan_actions = None
            
            for i, score in enumerate(plan_scores):
                if score is not None and i < len(plans) and score > best_score:
                    best_score = score
                    best_plan_actions = [int(a) for a in plans[i]["actions"]]
            
            # Store remaining actions from best plan
            if best_plan_actions:
                remaining_actions = best_plan_actions[1:]
                if remaining_actions:
                    self.action_buffer.set_plan(remaining_actions)
        
        # Add buffer status to the response
        vlm_json["buffer_status"] = self.action_buffer.get_buffer_status()
        
        return best_action, vlm_json

    # ---------------------------------------------------------------------
    # STEP 1 – prompt VLM, obtain JSON plan list --------------------------
    # ---------------------------------------------------------------------
    def check_plans(self, response_json):
        """
        Validate the 'plans' from the model response.

        - Ensures response is a dict and contains a 'plans' list
        - Each plan should have 'actions' of expected length (plan_horizon) and valid action values
        - The number of plans should match self.num_plans
        - If invalid, attempt to reprompt/recover up to 3 times
        Returns:
            List of valid plans (each plan is a dict with actions, explanation, expected_outcome), or None
        """
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            # 1. Verify response_json is a dict
            if not isinstance(response_json, dict):
                logger.warning(f"Response is not a dictionary: {type(response_json)}")
            else:
                # 2. Check for 'plans'
                plans = response_json.get("plans")
                if isinstance(plans, list) and len(plans) == self.num_plans:
                    valid = True
                    for idx, plan in enumerate(plans):
                        # 3. Check each plan dict and 'actions'
                        actions = plan.get("actions") if isinstance(plan, dict) else None
                        if (not isinstance(actions, list)) or (len(actions) != self.plan_horizon):
                            logger.warning(f"Plan {idx} has invalid or wrong number of actions: {actions}")
                            valid = False
                            break
                        for a in actions:
                            try:
                                ia = int(a)
                            except Exception:
                                logger.warning(f"Plan {idx} contains non-integer action: {a}")
                                valid = False
                                break
                            if not (0 <= ia < self.env.action_space.n):
                                logger.warning(f"Action {ia} in plan {idx} is out of valid range (0~{self.env.action_space.n-1})")
                                valid = False
                                break
                        # # Optionally check explanations/outcomes exist and are strings
                        # if not (isinstance(plan.get("explanation"), str) and isinstance(plan.get("expected_outcome"), str)):
                        #     logger.warning(f"Plan {idx} missing explanation or expected_outcome as strings")
                        #     valid = False
                        #     break
                    if valid:
                        return plans
                    # If we broke out due to invalid, retry below

                else:
                    logger.warning(f"'plans' missing or wrong length: {plans}")

            # Recovery: Reprompt for valid JSON structure, as in check_action
            error_message = (
                f"Your output format is invalid. Please provide exactly {self.num_plans} plans, "
                f"each with {self.plan_horizon} integer actions in a list under 'actions'. "
                f"All actions must be between 0 and {self.env.action_space.n-1}. "
                f"Each plan must include an 'expected_outcome' as strings. "
                "Format: {"
                '"reasoning": "...",'
                '"plans": ['
                '{"actions": [0, 1, ...], "expected_outcome": "..."}, ...'
                "]}"
            )
            self.add_user_message(user_msg=error_message)
            try:
                response = self.query_LLM()
                response_json = self.clean_response(response, self.path if hasattr(self, 'path') else "./")
                retry_count += 1
            except Exception as e:
                logger.error(f"Error getting new response (attempt {retry_count+1}): {str(e)}")
                retry_count += 1

        logger.error("Failed to get valid plans after multiple attempts, returning None")
        return None

    
    def _query_vlm_for_plans(self, path: str | Path) -> Dict[str, Any]:
        """Send image + instruction; parse & return JSON dict (may be empty)."""
        usr_msg = f"Analyze the current and historical game frames, decide the best action. Generate {self.num_plans} action plans of {self.plan_horizon} steps each in JSON format."
        
        frame = self.env.render()
        self.add_user_message(frame, usr_msg)
        response = self.get_response()  # from parent (auto retry etc.)
        #delete the usr_msg from messages but keep the frame
        self.delete_usr_message(usr_msg)

        if self.messages and self.messages[-1].get('role') == 'user' and self.messages[-1].get('content') == usr_msg:
            self.messages.pop()

        
        # Print the message history that will be sent to the VLM

        print("\n=== VLM Input Messages ===")
        for i, msg in enumerate(self.messages):
            role = msg.get('role', 'unknown')
            print(f"\nMessage {i+1} ({role}):")
            
            # Handle different message formats for different models
            if 'content' in msg:
                if isinstance(msg['content'], str):
                    print(f"  Content: {msg['content']}")
                elif isinstance(msg['content'], list):
                    for item in msg['content']:
                        if isinstance(item, dict):
                            if item.get('type') == 'text':
                                print(f"  Text: {item.get('text', '')}")
                            elif item.get('type') == 'image_url':
                                print("  Image: [image data]")
                        else:
                            print(f"  Content item: {item}")
            elif 'parts' in msg:
                for part in msg['parts']:
                    if 'text' in part:
                        print(f"  Text: {part['text']}")
                    elif 'mime_type' in part and 'data' in part:
                        print(f"  Image: [image data]")
            print("-" * 50)

        response = self.get_response()  # from parent (auto retry etc.)
        # if response:
        #     self.add_assistant_message()


        return self.clean_response(response, str(path)) # type: ignore[return-value]

    # ---------------------------------------------------------------------
    # STEP 2 – world‑model rollout to score each plan ---------------------
    # ---------------------------------------------------------------------

    def _pick_best_action_and_plan(self, rgb_obs: Union[np.ndarray, Image.Image], plans: List[Dict[str, Any]]) -> Tuple[int, int, List[float | None]]:
        """Return first action of plan with highest predicted cumulative reward, along with plan index and all scores."""
        try:
            if not self.use_world_model:
                return int(plans[0]['actions'][0]), 0, [None] * len(plans)
            
            # Convert PIL Image to numpy array if needed
            if hasattr(rgb_obs, 'convert'):
                rgb_obs = np.array(rgb_obs)
            
            # Fast resize using OpenCV
            if rgb_obs.shape[0] != 64 or rgb_obs.shape[1] != 64:
                rgb_obs = cv2.resize(rgb_obs, (64, 64), interpolation=cv2.INTER_AREA)
            
            # Convert to tensor and encode
            obs_tensor = _rgb_to_bchw(rgb_obs, self.device)
            with torch.no_grad():
                enc = self.tokenizer.encode(obs_tensor, should_preprocess=True)
                init_tokens = enc.tokens  # [1,K]
            
            # Cache token count if not already set
            if self._num_obs_tokens is None:
                self._num_obs_tokens = init_tokens.shape[1]
            
            # Parse actions from plans - do this more efficiently
            plan_actions = []
            for idx, plan in enumerate(plans):
                try:
                    # Convert actions to integers directly
                    actions = [int(a) for a in plan.get('actions', [])]
                    if len(actions) == self.plan_horizon:
                        plan_actions.append((idx, actions))
                except (ValueError, TypeError):
                    continue
            
            # If no valid plans, return default
            if not plan_actions:
                return int(plans[0]['actions'][0]), 0, [None] * len(plans)
                
            # Simulate all plans in parallel
            rewards = self._simulate_plans_parallel(init_tokens, plan_actions)
            
            # Find best plan more efficiently
            best_reward = float('-inf')
            best_idx = 0
            best_plan_idx = plan_actions[0][0]  # Default to first plan
            
            for i, (plan_idx, actions) in enumerate(plan_actions):
                reward = rewards[i]
                if reward is not None and reward > best_reward:
                    best_reward = reward
                    best_idx = i
                    best_plan_idx = plan_idx
            
            # Return the best action
            return int(plans[best_plan_idx]['actions'][0]), best_plan_idx, rewards
                
        except Exception as e:
            # Fallback to first plan on error
            return int(plans[0]['actions'][0]), 0, [None] * len(plans)

    def _pick_best_action(self, rgb_obs: Union[np.ndarray, Image.Image], plans: List[Dict[str, Any]]) -> int:
        """Legacy method for backward compatibility."""
        action, _, _ = self._pick_best_action_and_plan(rgb_obs, plans)
        return action

    def _simulate_plans_parallel(self, init_tokens: torch.LongTensor, plan_actions: List[Tuple[int, List[int]]]) -> List[Optional[float]]:
        """Simulate multiple plans in parallel and return their scores."""
        # Optimize for small number of plans - avoid overhead of thread creation for few plans
        if len(plan_actions) <= 2:
            return [self._simulate_safe(init_tokens, actions) for _, actions in plan_actions]
        
        # For more plans, use ThreadPoolExecutor with optimized worker count
        # Use at most 4 workers to prevent GPU thrashing
        with ThreadPoolExecutor(max_workers=min(len(plan_actions), 4)) as executor:
            # Submit all simulation tasks at once
            futures = [executor.submit(self._simulate_safe, init_tokens, actions) for _, actions in plan_actions]
            
            # Wait for all to complete and collect results
            return [future.result() if future.exception() is None else None for future in futures]
    
    def _simulate_safe(self, obs_tokens: torch.LongTensor, actions: List[int]) -> Optional[float]:
        """Wrapper for _simulate that catches exceptions."""
        try:
            return self._simulate(obs_tokens, actions)
        except Exception as e:
            logger.error(f"Error in simulation: {str(e)}")
            return None

    @torch.no_grad()
    def _simulate(self, obs_tokens: torch.LongTensor, actions: List[int]) -> float:
        """Return cumulative reward predicted by world model for a given plan."""
        tokens = obs_tokens.clone().to(self.device)
        cumulative_reward = 0.0
        done = False

        # Get max token limit from world model config if available
        max_tokens = None
        if hasattr(self.world_model, 'config'):
            if hasattr(self.world_model.config, 'max_blocks') and hasattr(self.world_model.config, 'tokens_per_block'):
                max_tokens = self.world_model.config.max_blocks * self.world_model.config.tokens_per_block
        
        # Cache the model for faster inference
        model = self.world_model.eval()
        
        # Process actions in batches when possible
        a_idx = 0
        while a_idx < len(actions) and not done:
            if max_tokens is not None and tokens.shape[1] >= max_tokens - 1:
                break
                
            try:
                # Create action token and concatenate with current tokens
                act_tok = torch.tensor([[actions[a_idx]]], device=self.device, dtype=torch.long)
                inp = torch.cat([tokens, act_tok], dim=1)
                
                # Run world model inference with no_grad already applied at function level
                out = model(inp)

                # Extract reward prediction
                if out.logits_rewards.size(1) > 0:
                    reward = out.logits_rewards.argmax(dim=-1).item() - 1  # logits class → {-1,0,1}
                    cumulative_reward += reward
                
                # Check if episode is done
                if out.logits_ends.size(1) > 0:
                    done = out.logits_ends.argmax(dim=-1).item() == 1
                
                # Get next observation tokens
                tokens = out.logits_observations.argmax(dim=-1)
                a_idx += 1
                
            except Exception as e:
                # Simplified error handling
                logger.error(f"Error during simulation at step {a_idx}: {str(e)}")
                break

        return cumulative_reward
    def delete_usr_message(self, usr_msg: str):
        """Delete the last user message if it matches the given text."""
        if self.messages and self.messages[-1].get('role') == 'user' and self.messages[-1].get('content')[0].get('text') == usr_msg:
            #onlt delete usr_message but keep the frame
            self.messages[-1]['content'] = self.messages[-1]['content'][1:]  # remove only the first part (the user message)

            logger.debug(f"Deleted user message: {usr_msg}")
        else:
            logger.debug("No matching user message to delete.")