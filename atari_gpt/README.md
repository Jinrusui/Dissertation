# atari_gpt

This is the official codebase for [atari_gpt](https://arxiv.org/pdf/2408.15950), a new benchmark for Large Language Models (LLMs) on Atari games. To see more results see our project [webpage](https://dev1nw.github.io/atari_gpt/).

## Set up

In order to run the code you need to have an API key for the respective model. To reproduce results from the paper you will need all 3 API keys.

&ensp;&ensp;&ensp;&ensp;For Google you can get an API key [here](https://ai.google.dev). Once you have this API key put it in a file called GOOGLE_API_KEY.txt.


&ensp;&ensp;&ensp;&ensp;For Anthropic you can get an API key [here](https://www.anthropic.com/api). Once you have this API key put it in a file called ANTHROPIC_API_KEY.txt.

&ensp;&ensp;&ensp;&ensp;For OpenAI you can get an API key [here](https://openai.com/api/). Once you have this API key put it in a file called OPENAI_API_KEY.txt.


## Installation

To run the code you will need to have Anaconda and run the following commands:

<br>&ensp;&ensp;&ensp;&ensp;`conda env create --file=environment.yaml`
<br>&ensp;&ensp;&ensp;&ensp;`conda activate atari_gpt`
<br>&ensp;&ensp;&ensp;&ensp;`python full_evaluation.py`

## Usage

The evaluation script supports several command-line arguments for specific models and environments but defaults to all models and environments:

```bash
python full_evaluation.py --models gpt4 claude --games PongDeterministic-v4 BreakoutDeterministic-v4
```

### Command-line Arguments

- `--models`: Specify which models to evaluate (default: all models)
  - Available models: `all`, `rand`, `gpt4`, `gpt4o`, `gemini`, `claude`
  - Example: `--models gpt4 claude`

- `--games`: Specify which games to evaluate (default: all games)
  - Example: `--games PongDeterministic-v4 BreakoutDeterministic-v4`

- `--output_dir`: Specify the directory to save experiment results (default: ./experiments)
  - Example: `--output_dir ./my_results`

## New: Analyzing Results

After running experiments, you can analyze and visualize the results using the `analyze_results.py` script:

```bash
python analyze_results.py --input_dir ./experiments --output_dir ./analysis
```

### Analysis Command-line Arguments

- `--input_dir`: Directory containing experiment results (default: ./experiments)
- `--output_dir`: Directory to save analysis results (default: ./analysis)
- `--models`: Models to include in analysis (default: all models)
- `--games`: Games to include in analysis (default: all games)

The script generates:
1. Reward curves for each game
2. A bar chart comparing final rewards across games and models
3. A JSON file with summary statistics

Example:
```bash
python analyze_results.py --models gpt4 claude --games Pong Breakout
```

## Citing atari_gpt 

```
@misc{waytowich2024atarigptinvestigatingcapabilitiesmultimodal,
      title={atari_gpt: Investigating the Capabilities of Multimodal Large Language Models as Low-Level Policies for Atari Games}, 
      author={Nicholas R. Waytowich and Devin White and MD Sunbeam and Vinicius G. Goecks},
      year={2024},
      eprint={2408.15950},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2408.15950}, 
}
```