# Dissertation

This repository contains the code and resources related to my dissertation project on Vision Language Models (VLMs) and World Models for reinforcement learning.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview

This dissertation explores the application of Vision Language Models (VLMs) in world modeling for reinforcement learning environments. The research focuses on how VLMs can be leveraged to create more effective world models that understand and predict complex environments, particularly in visual domains like Atari games. The repository contains implementations of various approaches, including:

- VLM-based world models for environment prediction
- Iris models for visual understanding
- Atari-GPT for game state prediction and policy learning

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Jinrusui/Dissertation.git
   cd Dissertation
   ```

2. CUDA Requirements:
   This project requires CUDA 11.3.0 (included in the repository as `cuda_11.3.0_465.19.01_linux.run`). To install CUDA:
   ```bash
   chmod +x cuda_11.3.0_465.19.01_linux.run
   sudo ./cuda_11.3.0_465.19.01_linux.run
   ```
   Follow the installation prompts. Make sure to install the CUDA Toolkit and appropriate drivers.

3. Create and activate the conda environment:
   - For Linux/macOS:
     ```bash
     conda env create -f environment.yaml
     conda activate vlm_wm
     ```
   - For Windows:
     ```bash
     conda env create -f environment_win.yaml
     conda activate vlm_wm
     ```

## Usage

### Running VLM World Model Experiments

Navigate to the vlm_world_model directory and run the main script:
```bash
cd vlm_world_model
python src/main.py --config=config/default.yaml
```


### Atari-GPT Experiments

For running Atari game experiments with GPT integration:
```bash
cd atari_gpt
python run_experiments.py --game=Breakout --model=gpt4
```

## Project Structure

- `vlm_world_model/` – Implementation of VLM-based world models
  - `src/` – Source code for the world model implementation
  - `config/` – Configuration files for experiments
  
- `iris/` – Vision model implementation
  - `models/` – Model architecture definitions
  - `data/` – Data processing utilities
  - `checkpoints/` – Model checkpoints (not included in git)

- `atari_gpt/` – Integration of GPT models with Atari environments
  - `envs/` – Environment wrappers
  - `models/` – Policy and value network implementations

- `outputs/` – Experiment outputs and results (not included in git)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or feedback, contact [Jinrusui](https://github.com/Jinrusui).
