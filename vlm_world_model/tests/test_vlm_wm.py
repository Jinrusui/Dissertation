"""Unit tests for VLMWorldModelAgent."""

import os
import sys
from pathlib import Path

# Add both project root and parent directory to Python path
project_root = Path(__file__).parent.parent
parent_dir = project_root.parent
sys.path.insert(0, str(parent_dir))  # Add parent directory first
sys.path.insert(0, str(project_root))  # Then add project root

import unittest
from unittest.mock import Mock, patch
import numpy as np
import torch

from vlm_world_model.src.vlm_wm import VLMWorldModelAgent  # Updated import path
from iris.src.models.tokenizer.tokenizer import Tokenizer
from iris.src.models.world_model import WorldModel

class TestVLMWorldModelAgent(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Mock the parent Agent class
        self.mock_agent = Mock()
        self.mock_agent.get_response.return_value = "mock response"
        self.mock_agent.clean_response.return_value = {
            "reasoning": "test reasoning",
            "plans": [
                {
                    "actions": [0, 1, 2, 3, 4],
                    "explanation": "test explanation",
                    "expected_outcome": "test outcome"
                },
                {
                    "actions": [1, 2, 3, 4, 5],
                    "explanation": "test explanation 2",
                    "expected_outcome": "test outcome 2"
                },
                {
                    "actions": [2, 3, 4, 5, 6],
                    "explanation": "test explanation 3",
                    "expected_outcome": "test outcome 3"
                }
            ]
        }

        # Mock environment
        self.mock_env = Mock()
        self.mock_env.action_space.n = 18
        self.mock_env.render.return_value = np.zeros((84, 84, 3), dtype=np.uint8)

        # Mock tokenizer
        self.mock_tokenizer = Mock(spec=Tokenizer)
        self.mock_tokenizer.encode.return_value = Mock(
            tokens=torch.zeros((1, 10), dtype=torch.long)
        )

        # Mock world model
        self.mock_world_model = Mock(spec=WorldModel)
        mock_output = Mock()
        mock_output.logits_rewards = torch.tensor([[[0.1, 0.8, 0.1]]])  # [1,1,3]
        mock_output.logits_ends = torch.tensor([[[0.9, 0.1]]])  # [1,1,2]
        mock_output.logits_observations = torch.zeros((1, 1, 10))  # [1,1,10]
        self.mock_world_model.return_value = mock_output

        # Create agent instance
        self.agent = VLMWorldModelAgent(
            tokenizer=self.mock_tokenizer,
            world_model=self.mock_world_model,
            plan_horizon=5,
            num_plans=3,
            device="cpu",
            env=self.mock_env
        )
        self.agent.get_response = self.mock_agent.get_response
        self.agent.clean_response = self.mock_agent.clean_response

    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.plan_horizon, 5)
        self.assertEqual(self.agent.num_plans, 3)
        self.assertEqual(self.agent.device, torch.device("cpu"))
        self.assertIsNone(self.agent._num_obs_tokens)

    def test_check_plans_valid(self):
        """Test check_plans with valid input."""
        valid_json = {
            "reasoning": "test",
            "plans": [
                {
                    "actions": [0, 1, 2, 3, 4],
                    "explanation": "test",
                    "expected_outcome": "test"
                },
                {
                    "actions": [1, 2, 3, 4, 5],
                    "explanation": "test",
                    "expected_outcome": "test"
                },
                {
                    "actions": [2, 3, 4, 5, 6],
                    "explanation": "test",
                    "expected_outcome": "test"
                }
            ]
        }
        plans = self.agent.check_plans(valid_json)
        self.assertIsNotNone(plans)
        self.assertEqual(len(plans), 3)

    def test_check_plans_invalid(self):
        """Test check_plans with invalid input."""
        invalid_json = {
            "reasoning": "test",
            "plans": [
                {
                    "actions": [0, 1, 2],  # Wrong length
                    "explanation": "test",
                    "expected_outcome": "test"
                }
            ]
        }
        plans = self.agent.check_plans(invalid_json)
        self.assertIsNone(plans)

    def test_pick_best_action(self):
        """Test _pick_best_action method."""
        rgb_obs = np.zeros((84, 84, 3), dtype=np.uint8)
        plans = [
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6]
        ]
        action = self.agent._pick_best_action(rgb_obs, plans)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < self.mock_env.action_space.n)

    def test_simulate(self):
        """Test _simulate method."""
        obs_tokens = torch.zeros((1, 10), dtype=torch.long)
        actions = [0, 1, 2, 3, 4]
        reward = self.agent._simulate(obs_tokens, actions)
        self.assertIsInstance(reward, float)

    def test_generate_response(self):
        """Test generate_response method."""
        path = Path("test_path")
        action, response = self.agent.generate_response(path)
        self.assertIsInstance(action, int)
        self.assertIsInstance(response, dict)
        self.assertTrue(0 <= action < self.mock_env.action_space.n)

if __name__ == "__main__":
    unittest.main()