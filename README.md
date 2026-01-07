# rogue-gym-agents-cog19
This repository contains codes for exprimens in
[COG2019 Rogue-Gym: A New Challenge for Generalization in Reinforcement Learning](https://arxiv.org/abs/1904.08129).

## Watch agents
Here I show gifs of agents appeared in Section 6.4 in the paper.

### Overfitted Agent
CNN + 10 training seeds

![Overfitted](pictures/overfitted.gif)

### Generalized Agent
ResNet + L2 regularization + 40 training seeds

![Generalized](pictures/generalized.gif)

## Setup
1. Create and activate virtual environment
```bash
python3 -m venv venv38
source venv38/bin/activate
```

2. Install dependencies
```bash
pip install -r requirements.txt
# Or install manually:
# pip install torch numpy gym rogue-gym rainy
```

## Usage
All hyper paremters are at [env.py](agents/env.py) and you need to
edit the file to change the experiment setting.

### Train agents
- PPO with nature CNN
```bash
python agents/ppo_naturecnn.py train
```

- PPO with impala CNN
```bash
python agents/ppo_impalacnn.py train
```

- PPO with β-VAE feature extractor(β is hard coded in the file)
```bash
python agents/vae_ppo.py train
```

### Evaluate agents
```bash
python agents/eval_seeds.py --logdir=$YOUR_LOD_DIR
```

