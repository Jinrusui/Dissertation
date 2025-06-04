"""VLM‑WorldModel Agent
====================
A drop‑in replacement for the existing *Agent* class that:
1.  Lets an image‑VLM (GPT‑4o / Gemini / Claude‑Vision …) **propose multiple multi‑step action plans**.
2.  Evaluates every plan inside an **offline world‑model** (tokenizer + world_model) to obtain cumulative reward.
3.  Selects the plan with the best predicted reward and returns **its first action** for real‑environment execution.

It keeps **exactly** the same public interface as the old `Agent` used in *run_experiments.py*:
```python
agent = VLMWorldModelAgent(model_name=..., model="gpt4o", system_message=..., env=env,
                           tokenizer=tok, world_model=wm, device="cuda:0")
action, full_response = agent.generate_response(path)   # unchanged call‑site
```
– `full_response` is now the raw JSON produced by the VLM.

The new agent subclasses the original `Agent` and **only overrides the upper‑level call‑chain** so other methods in *run_experiments* stay intact.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import sys
sys.path.insert(0, '/mnt/e/Projects/Dissertation')

from atari_gpt.llms import Agent  # parent base class (already on import path)
from iris.src.models.tokenizer.tokenizer import Tokenizer, TokenizerEncoderOutput
from iris.src.models.world_model import WorldModel, WorldModelOutput

logger = logging.getLogger("vlm_wm_agent")

__all__ = ["VLMWorldModelAgent"]


def _rgb_to_bchw(rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    """H × W × C uint8 → 1 × C × H × W float32[0,1]."""
    if rgb.dtype != np.uint8:
        rgb = (rgb.clip(0, 1) * 255).astype(np.uint8)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
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
        **agent_kwargs: Any,
    ) -> None:
        super().__init__(**agent_kwargs)  # initialise LLM client etc.

        self.device = torch.device(device)
        self.tokenizer = tokenizer.to(self.device).eval()
        self.world_model = world_model.to(self.device).eval()

        self.plan_horizon = plan_horizon
        self.num_plans = num_plans

        self._num_obs_tokens: int | None = None  # cache after first encode

    # ---------------------------------------------------------------------
    # PUBLIC API (unchanged signature) ------------------------------------
    # ---------------------------------------------------------------------

    def generate_response(self, path: str | Path) -> Tuple[int, Dict[str, Any]]:  # type: ignore[override]
        """Overrides parent: returns *(selected_action, raw_VLM_JSON)*.

        * path is still required because run_experiments writes raw completions there.
        * We store VLM raw output exactly like parent; then we compute best‑plan action.
        """
        # === 1. Ask VLM for plans ===================================================
        vlm_json = self._query_vlm_for_plans(path)
        plans = self.check_plans(vlm_json)
        if not plans:
            logger.warning("No valid plans parsed – fallback to NOOP")
            return 0, vlm_json

        # === 2. Evaluate via world‑model ============================================
        obs_rgb = self.env.render(mode="rgb_array")
        best_action = self._pick_best_action(obs_rgb, plans)
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
                        # Optionally check explanations/outcomes exist and are strings
                        if not (isinstance(plan.get("explanation"), str) and isinstance(plan.get("expected_outcome"), str)):
                            logger.warning(f"Plan {idx} missing explanation or expected_outcome as strings")
                            valid = False
                            break
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
                f"Each plan must include an 'explanation' and 'expected_outcome' as strings. "
                "Format: {"
                '"reasoning": "...",'
                '"plans": ['
                '{"actions": [0, 1, ...], "explanation": "...", "expected_outcome": "..."}, ...'
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
            "Generate {num_plans} alternative action plans of {plan_horizon} steps each "
            "following the required JSON format.".format(num_plans=self.num_plans, plan_horizon=self.plan_horizon)
        )
        print(usr_msg)
        frame = self.env.render()
        self.add_user_message(frame, usr_msg)

        response = self.get_response()  # from parent (auto retry etc.)
        json_obj = self.clean_response(response, str(path))
        print("\n\nresponse_json:", json_obj)
        return json_obj # type: ignore[return-value]

    # ---------------------------------------------------------------------
    # STEP 2 – world‑model rollout to score each plan ---------------------
    # ---------------------------------------------------------------------

    def _pick_best_action(self, rgb_obs: np.ndarray, plans: List[List[int]]) -> int:
        """Return first action of plan with highest predicted cumulative reward."""
        obs_tensor = _rgb_to_bchw(rgb_obs, self.device)
        enc: TokenizerEncoderOutput = self.tokenizer.encode(obs_tensor, should_preprocess=True)
        init_tokens = enc.tokens  # [1,K]
        if self._num_obs_tokens is None:
            self._num_obs_tokens = init_tokens.shape[1]

        best_score = -float("inf")
        best_first_action = 0
        for plan in plans:
            score = self._simulate(init_tokens, plan)
            if score > best_score:
                best_score, best_first_action = score, plan[0]
        return int(best_first_action)

    @torch.no_grad()
    def _simulate(self, obs_tokens: torch.LongTensor, actions: List[int]) -> float:
        """Return cumulative reward predicted by world model for a given plan."""
        tokens = obs_tokens.clone().to(self.device)
        cumulative_reward = 0.0
        done = False

        for a in actions:
            if done:
                break
            act_tok = torch.tensor([[a]], device=self.device, dtype=torch.long)
            inp = torch.cat([tokens, act_tok], dim=1)
            out: WorldModelOutput = self.world_model(inp)

            reward = out.logits_rewards.argmax(dim=-1).item() - 1  # logits class → {-1,0,1}
            cumulative_reward += reward
            done = out.logits_ends.argmax(dim=-1).item() == 1
            tokens = out.logits_observations.argmax(dim=-1)

        return cumulative_reward
