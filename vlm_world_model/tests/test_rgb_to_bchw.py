#!/usr/bin/env python3
"""
RGB to BCHW Conversion Test
==========================
This script tests the _rgb_to_bchw function with different input types
to ensure it handles all cases correctly.
"""

import sys
import os
import numpy as np
import torch
import gymnasium as gym
import cv2
from PIL import Image
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.vlm_wm import _rgb_to_bchw

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("rgb_to_bchw_test")

def test_numpy_uint8():
    """Test with numpy uint8 array (HWC format)."""
    logger.info("Testing with numpy uint8 array (HWC format)")
    rgb = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    try:
        tensor = _rgb_to_bchw(rgb, torch.device("cpu"))
        logger.info(f"Success! Output shape: {tensor.shape}, dtype: {tensor.dtype}, range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
        return True
    except Exception as e:
        logger.error(f"Failed: {str(e)}")
        return False

def test_numpy_float():
    """Test with numpy float array (HWC format)."""
    logger.info("Testing with numpy float array (HWC format)")
    rgb = np.random.random((64, 64, 3)).astype(np.float32)
    try:
        tensor = _rgb_to_bchw(rgb, torch.device("cpu"))
        logger.info(f"Success! Output shape: {tensor.shape}, dtype: {tensor.dtype}, range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
        return True
    except Exception as e:
        logger.error(f"Failed: {str(e)}")
        return False

def test_numpy_chw():
    """Test with numpy array in CHW format."""
    logger.info("Testing with numpy array (CHW format)")
    rgb = np.random.randint(0, 255, (3, 64, 64), dtype=np.uint8)
    try:
        tensor = _rgb_to_bchw(rgb, torch.device("cpu"))
        logger.info(f"Success! Output shape: {tensor.shape}, dtype: {tensor.dtype}, range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
        return True
    except Exception as e:
        logger.error(f"Failed: {str(e)}")
        return False

def test_pil_image():
    """Test with PIL Image."""
    logger.info("Testing with PIL Image")
    rgb = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    pil_img = Image.fromarray(rgb)
    try:
        # Convert PIL Image to numpy array first
        rgb_array = np.array(pil_img)
        tensor = _rgb_to_bchw(rgb_array, torch.device("cpu"))
        logger.info(f"Success! Output shape: {tensor.shape}, dtype: {tensor.dtype}, range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
        return True
    except Exception as e:
        logger.error(f"Failed: {str(e)}")
        return False

def test_string():
    """Test with string (should fail gracefully)."""
    logger.info("Testing with string (should fail gracefully)")
    try:
        tensor = _rgb_to_bchw("this is a test string", torch.device("cpu"))
        logger.error("Failed: String was accepted but should have been rejected")
        return False
    except TypeError as e:
        logger.info(f"Success! Correctly rejected string with error: {str(e)}")
        return True
    except Exception as e:
        logger.error(f"Failed with unexpected error: {str(e)}")
        return False

def test_gym_env():
    """Test with actual gym environment observation."""
    logger.info("Testing with gym environment observation")
    try:
        env = gym.make("Breakout-v4", render_mode="rgb_array")
        observation, _ = env.reset()
        tensor = _rgb_to_bchw(observation, torch.device("cpu"))
        logger.info(f"Success! Output shape: {tensor.shape}, dtype: {tensor.dtype}, range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
        env.close()
        return True
    except Exception as e:
        logger.error(f"Failed: {str(e)}")
        return False

def test_resized_observation():
    """Test with resized gym environment observation."""
    logger.info("Testing with resized gym environment observation")
    try:
        env = gym.make("Breakout-v4", render_mode="rgb_array")
        observation, _ = env.reset()
        resized = cv2.resize(observation, (512, 512))
        tensor = _rgb_to_bchw(resized, torch.device("cpu"))
        logger.info(f"Success! Output shape: {tensor.shape}, dtype: {tensor.dtype}, range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
        env.close()
        return True
    except Exception as e:
        logger.error(f"Failed: {str(e)}")
        return False

def main():
    """Run all tests."""
    tests = [
        test_numpy_uint8,
        test_numpy_float,
        test_numpy_chw,
        test_pil_image,
        test_string,
        test_gym_env,
        test_resized_observation
    ]
    
    results = []
    for test_fn in tests:
        results.append(test_fn())
    
    # Print summary
    logger.info("\n=== Test Summary ===")
    for i, (test_fn, result) in enumerate(zip(tests, results)):
        status = "PASSED" if result else "FAILED"
        logger.info(f"{i+1}. {test_fn.__name__}: {status}")
    
    success_rate = sum(results) / len(results) * 100
    logger.info(f"Success rate: {success_rate:.1f}% ({sum(results)}/{len(results)})")
    
    if all(results):
        logger.info("All tests passed!")
    else:
        logger.warning("Some tests failed!")

if __name__ == "__main__":
    main() 