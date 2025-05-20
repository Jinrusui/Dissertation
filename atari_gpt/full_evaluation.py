import json
import argparse
import logging
import os
import sys
from run_experiments import run

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)

# Suppress httpx and urllib3 logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger("atari_gpt")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run atari_gpt evaluations')
    parser.add_argument('--models', nargs='+', default=['rand', 'gpt4', 'gpt4o', 'gemini', 'claude','qwen'],
                        help='Models to evaluate (default: all models)')
    parser.add_argument('--games', nargs='+', default=None,
                        help='Specific games to evaluate (default: all games)')
    parser.add_argument('--output_dir', default='./experiments',
                        help='Directory to save experiment results')
    args = parser.parse_args()
    
    # If 'all' is specified, replace it with the list of all models
    if 'all' in args.models:
        args.models = ['rand', 'gpt4', 'gpt4o', 'gemini', 'claude']
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load environment configurations
    try:
        with open('envs.json') as file:
            environment_list = json.load(file)
    except FileNotFoundError:
        logger.error("envs.json file not found. Please make sure it exists in the current directory.")
    except json.JSONDecodeError:
        logger.error("Error parsing envs.json. Please check the file format.")
    
    # Filter games if specified
    if args.games:
        environments = {k: v for k, v in environment_list.items() if k in args.games}
        if not environments:
            logger.error(f"None of the specified games {args.games} found in envs.json")
    else:
        environments = environment_list
    
    # Run evaluations for each model and game
    for model in args.models:
        logger.info(f"Starting evaluation for model: {model}")
        for game, prompt in environments.items():
            try:
                logger.info(f"Running test for: {game}")
                results = run(game, prompt, model, output_dir=args.output_dir)
                logger.info(f"Completed test for: {game}")
            except Exception as e:
                logger.error(f"Error running test for {game} with model {model}: {str(e)}")
                continue
        logger.info(f"Completed evaluation for model: {model}")

if __name__ == "__main__":
    main()
    
