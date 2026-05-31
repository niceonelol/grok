# OpenAI Grok Curve Experiments

## Paper

This is the code for the paper [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177) by Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, and Vedant Misra

## Installation and Training

Upon cloning, run the following in terminal, to ensure the git submodules are also cloned:

```bash
git submodule update --init --recursive
```

The code for this program runs on Python 3.12.10 (incompatible with 3.13.x or later) so please create a virtual environment on this version of Python to run the code

In phd/topology.py at the start of the calculate_ph_dim_gpu function, the import shown is:

```bash
from torchph.pershom import vr_persistence 
```

This should be renamed to:

```bash
from torchph.torchph.pershom import vr_persistence
```

In your virtual environment, run the following, ensuring you are in the root directory of the project:

```bash
pip install -r requirements.txt
pip install -e .
```