## This is the official repository of MUSIC (Multitask Learning Under Sparse and Incomplete Constraints)

This repository contains four folders, one for each of the coupled systems in the paper. Each folder contains multiple files that include a python notebook containing the model training, a sample trained model that can be directly used to replicate some results in the paper, and any relevant datatset (if applicable).

If you want to run the training, please directly download the folder of your choice and run the respective .ipynb notebook. 

Note that the training and validation splits are random and so the results might vary slightly from the numbers reported in the paper.

The details of compute resources used in training of these models along with time taken for each of the models is given below. 

Computation resources: The authors used the Trillium Cluster under Digital Research Alliance of Canada (formerly known as Compute Canada). For some initial simulations, evaluations and plot generation, Google Colab was also used.

Computational time: 
### SWE System

| Neurons \ Layers | 2    | 4    | 6    |
|------------------|-----:|-----:|-----:|
| 20               | 81.6 | 107  | 133.4|
| 50               | 82   | 108  | 136  |

The table above depicts the training time (in seconds) for the SWE system for different model complexities.

### FN System
| Neurons \ Layers | 4    | 6    |
|------------------|-----:|-----:|
| 100              | 2104 | 6670 |
| 200              | 4576 | 15000|

The table above depicts the training time (in seconds) for the FN system for different model complexities.

### RD System
| Neurons \ Layers | 4    | 6    |
|------------------|-----:|-----:|
| 50               | 3311 | 6092 |
| 100              | 6108 | 9300 |
The table above depicts the training time (in seconds) for the RD system for different model complexities.

### Physical System of Wildfire
| Neurons \ Layers | 4    | 6    |
|------------------|-----:|-----:|
| 50               | 3073 | 3181 |
| 100              | 3089 | 3172 |

The table above depicts the training time (in seconds) for the physical system of wildfire for different model complexities. Note the trainng time doesn't differ much as we are using IHT and not the neuron $\ell_0$.




