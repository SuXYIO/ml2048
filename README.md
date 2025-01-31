# 2048ml

## Intro

A ML experiment on the game 2048

## Components

The project consists of three main files
- `network.py` neural network definitions, and template export
- `traindqn.py` use DQN to train
- `trainevo.py` use evolutionary algorithm to train
- `evalmodel.py` evaluate trained model

And a package
- `Game2048Env` gymnasium env for game 2048

## Installation

For the gymnasium to work, you have to run

```bash
pip install -e Game2048Env
```

## Usage

So, training a network might look like this

```bash
# Export the template
python3 network.py fnn0 templates/fnn0.pth
# Train the model via DQN
python3 traindqn.py templates/fnn0.pth saves/fnn0_256e.pth 256
# Evaluate the model score
python3 evalmodel.py
```

Well, all the files are built with `argparse`, so you can check detailed usage via `python3 foo.py -h`.

## Todo

- [ ] Add more arguments for `trainevo.py`
- [ ] Add dynamic plots for training
- [ ] Find better hyperparameters
- [ ] Remove the redundant template stuff
- [ ] Add human render mode for env

