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
        
    if hasattr(rgb, 'dtype') and rgb.dtype != np.uint8:
        # If we have float values, ensure they're in [0,1] and convert to uint8
        rgb = (rgb.clip(0, 1) * 255).astype(np.uint8)
    
    # Handle different channel arrangements
    if len(rgb.shape) == 3 and rgb.shape[2] == 3:
        # Standard HWC format
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    elif len(rgb.shape) == 3 and rgb.shape[0] == 3:
        # Already in CHW format
        tensor = torch.from_numpy(rgb).float().div(255.0).unsqueeze(0)
    else:
        raise ValueError(f"Unexpected image shape: {rgb.shape}")
    
    return tensor.to(device)


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
        # First check if there's an action in the buffer
        buffered_action = self.action_buffer.get_action()
        if buffered_action is not None:
            logger.info(f"Using action {buffered_action} from buffer")
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
            logger.warning("No valid plans parsed – fallback to NOOP")
            return 0, vlm_json

        # === 2. Evaluate via world‑model ============================================
        obs_rgb = self.env.render()
        best_action, best_plan_idx, plan_scores = self._pick_best_action_and_plan(obs_rgb, plans)
        
        # === 3. Store the best plan in the buffer =================================
        if self.use_action_buffer and plans:
            # Create validated plans with scores
            validated_plans = []
            for i, score in enumerate(plan_scores):
                if score is not None and i < len(plans):
                    plan_actions = [int(a) for a in plans[i]["actions"]]
                    validated_plans.append((score, plan_actions))
            
            if validated_plans:
                # Sort plans by score to find the best one
                validated_plans.sort(key=lambda x: x[0], reverse=True)
                best_score, best_plan = validated_plans[0]
                
                # Get the remaining actions of the best plan
                remaining_plan = best_plan[1:]
                
                if remaining_plan:
                    # Set the single best plan in the buffer, replacing any old one
                    self.action_buffer.set_plan(remaining_plan)
        
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
        usr_msg = (
            "Analyze the current game frame, focus on ball and paddle position, and history of the message, decide the best action to take."
            "Generate {num_plans} alternative action plans of {plan_horizon} steps each "
            "following the required JSON format.".format(num_plans=self.num_plans, plan_horizon=self.plan_horizon)
        )
        #print(usr_msg)
        frame = self.env.render()
        self.add_user_message(frame, usr_msg)

        response = self.get_response()  # from parent (auto retry etc.)
        json_obj = self.clean_response(response, str(path))
        print("\n\nresponse_json:", json_obj)
        return json_obj # type: ignore[return-value]

    # ---------------------------------------------------------------------
    # STEP 2 – world‑model rollout to score each plan ---------------------
    # ---------------------------------------------------------------------

    def _pick_best_action_and_plan(self, rgb_obs: Union[np.ndarray, Image.Image], plans: List[Dict[str, Any]]) -> Tuple[int, int, List[float | None]]:
        """Return first action of plan with highest predicted cumulative reward, along with plan index and all scores."""
        try:
            if not self.use_world_model:
                return plans[0]['actions'][0], 0, [None] * len(plans)
            else:
                # Convert PIL Image to numpy array if needed
                if hasattr(rgb_obs, 'convert'):  # Check if it's a PIL Image
                    rgb_obs = np.array(rgb_obs)
                
                # Resize observation if needed to match tokenizer's expected input size
                if rgb_obs.shape[0] != 64 or rgb_obs.shape[1] != 64:
                    logger.info(f"Resizing observation from {rgb_obs.shape} to 64x64")
                    rgb_obs = cv2.resize(rgb_obs, (64, 64))
                
                obs_tensor = _rgb_to_bchw(rgb_obs, self.device)
                enc: TokenizerEncoderOutput = self.tokenizer.encode(obs_tensor, should_preprocess=True)
                init_tokens = enc.tokens  # [1,K]
                
                if self._num_obs_tokens is None:
                    self._num_obs_tokens = init_tokens.shape[1]
                
                # Parse actions from each plan
                plan_actions = []
                for idx, plan in enumerate(plans):
                    try:
                        actions = [int(a) for a in plan['actions']]
                        plan_actions.append((idx, actions))
                    except (ValueError, KeyError) as e:
                        logger.error(f"Error parsing actions from plan {idx}: {e}")
                
                # Simulate all plans in parallel
                rewards = self._simulate_plans_parallel(init_tokens, plan_actions)
                
                # Find best plan
                best_reward = float('-inf')
                best_idx = 0
                for idx, reward in enumerate(rewards):
                    if reward is not None and reward > best_reward:
                        best_reward = reward
                        best_idx = idx
                
                if best_idx < len(plan_actions):
                    plan_idx, actions = plan_actions[best_idx]
                    logger.info(f"Best plan {plan_idx} with reward {best_reward:.2f}")
                    return actions[0], plan_idx, rewards
                else:
                    logger.warning(f"Invalid best_idx {best_idx}, using plan 0")
                    return plans[0]['actions'][0], 0, rewards
                
        except Exception as e:
            logger.error(f"Error in _pick_best_action_and_plan: {e}")
            return plans[0]['actions'][0], 0, [None] * len(plans)

    def _pick_best_action(self, rgb_obs: Union[np.ndarray, Image.Image], plans: List[Dict[str, Any]]) -> int:
        """Legacy method for backward compatibility."""
        action, _, _ = self._pick_best_action_and_plan(rgb_obs, plans)
        return action

    def _simulate_plans_parallel(self, init_tokens: torch.LongTensor, plan_actions: List[Tuple[int, List[int]]]) -> List[Optional[float]]:
        """Simulate multiple plans in parallel and return their scores."""
        # Use ThreadPoolExecutor for parallelization
        with ThreadPoolExecutor(max_workers=min(len(plan_actions), mp.cpu_count())) as executor:
            # Submit simulation tasks
            futures = []
            for _, actions in plan_actions:
                futures.append(executor.submit(self._simulate_safe, init_tokens, actions))
            
            # Collect results
            scores = []
            for future in futures:
                try:
                    scores.append(future.result())
                except Exception as e:
                    logger.error(f"Error in parallel simulation: {str(e)}")
                    scores.append(None)
            
            return scores
    
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
                logger.info(f"Max tokens: {max_tokens}, Current tokens: {tokens.shape[1]}")

        for a_idx, a in enumerate(actions):
            if done:
                logger.debug(f"Stopping simulation at step {a_idx} due to done=True")
                break
            
            # Check if we've reached the token limit
            if max_tokens is not None and tokens.shape[1] >= max_tokens - 1:
                logger.warning(f"Reached token limit during simulation at step {a_idx}, stopping early")
                break
            
            try:
                # Create action token and concatenate with current tokens
                act_tok = torch.tensor([[a]], device=self.device, dtype=torch.long)
                inp = torch.cat([tokens, act_tok], dim=1)
                
                # Run world model inference
                out = self.world_model(inp)

                # Extract reward prediction
                if out.logits_rewards.size(1) > 0:
                    reward = out.logits_rewards.argmax(dim=-1).item() - 1  # logits class → {-1,0,1}
                    cumulative_reward += reward
                else:
                    logger.warning(f"Empty rewards tensor at step {a_idx}")
                
                # Check if episode is done
                if out.logits_ends.size(1) > 0:
                    done = out.logits_ends.argmax(dim=-1).item() == 1
                
                # Get next observation tokens
                tokens = out.logits_observations.argmax(dim=-1)
                
            except IndexError as e:
                logger.error(f"Index error during simulation at step {a_idx}: {str(e)}")
                break
            except RuntimeError as e:
                logger.error(f"Runtime error during simulation at step {a_idx}: {str(e)}")
                break
            except Exception as e:
                logger.error(f"Error during simulation at step {a_idx}: {str(e)}")
                break

        return cumulative_reward
