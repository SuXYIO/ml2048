# 2048ml

## Intro

A ML experiment on the game 2048

## Installation

Like most python projects

```bash
pip install -r requirements.txt
```

For the 2048 gymnasium env to work, you have to run

```bash
pip install -e Game2048Env
```

in this directory

## Usage

So, simple usage might look like this

```bash
# Export the template
python3 network.py fnn0 templates/fnn0.pth
# Train the model via DQN
python3 traindqn.py templates/fnn0.pth saves/fnn0_256e.pth 256
# Evaluate the model score
python3 evalmodel.py
# See the network run by yourself
python3 demo.py saves/fnn0_256e.pth
```

Well, all the files are built with `argparse`, so you can check **detailed usage** via `python3 foo.py -h`.

## Components

The project consists of three main files

| file | desc |
| ---- | ---- |
| `network.py` | neural network **definitions**, and template **export** |
| `traindqn.py` | use DQN to **train** |
| `trainevo.py` | use evolutionary algorithm to **train** |
| `evalmodel.py` | **evaluate** trained model |
| `demo.py` | **see** the trained model work |

And a package

| package | desc |
| ------- | ---- |
| `Game2048Env` | gymnasium env for game 2048 |

## Dependencies

| name | note |
| ---- | ---- |
| `torch` | / |
| `matplotlib` | / |
| `numpy` | / |
| `gymnasium` | / |
| `evotorch` | only necessary for `trainevo.py` |

## Todo

- [ ] Add more **arguments**
- [ ] Add **dynamic plots** for training
- [ ] Find better **hyperparameters**
- [ ] Remove the redundant **template** stuff

Open and **glad** for *PR*s.  

## Notes

This is just a little experiment, so im only using a `main` branch, at least until i get this to work as a big project, which is unlikely.  
That means this repo is **really unstable**, so better *fork* it if you wanna use it yourself.  

