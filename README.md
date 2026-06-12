# Arya Narang's code for Imperial Dissertation titled: Topology of Grokking

As mentioned clearly in the 'Declarations' section of the main report of my dissertation, this code is forked from the repo of Power et al. [1]: https://github.com/openai/grok

Further information on code which is/isn't written by me can be found in that section of the report (e.g. the Git submodules in this code are cloned from the papers of [6, 31, 32]). Furthermore, I will state in specific files if code is written by someone else, along with the appropriate reference (refer to my main report to find the references). Otherwise, the remainder of this README will focus on how to run the code and how to access the data.

## Accessing data

Graphs included in Chapter 3. 4 & 5.3 of my dissertation can be found in fyp/data. Note that there are more graphs here than those included in the paper as other experiments were conducted that are not included. These also include the relevant CSV files and models that were logged/trained. 

Data for Chapter 5.1 & 5.2 of my dissertation can be found in sanity_checker/e_alpha_data and sanity_checker/phdim_data respectively. 

## Training MLPs on MNIST

Here is an example on how to run the code from Chapter 4 for a training size of 2000, a weights scale factor of 4.0 on the random seed 47:

```
py mnist-grok/model.py --train_size 2000 --scale_factor 4.0 --seed 47
```

The snippet above will train that model for 100,000 epochs.

## Setting up

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
