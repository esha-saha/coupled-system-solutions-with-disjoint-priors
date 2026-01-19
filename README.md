# This is the official repository of MUSIC (Multitask Learning Under Sparse and Incomplete Constraints)

The corresponding paper title "Learning Coupled System Dynamics under Incomplete Physical Constraints and Missing Data" can be found at https://arxiv.org/pdf/2512.23761.

## Problem Statement
Let $\Omega\subset\mathbb{R}^d$ be a spatial domain and $T>0$ be a final time. Our goal is to learn the solutions $\mathbf{u} = \{u_1,u_2,\dots,u_n\}$, $\mathbf{u}:\Omega\times (0,T]\rightarrow\mathbb{R}^n$ governed by a set of $n$ partial differential equations (PDEs) i.e.,

$\dfrac{\partial u_i}{\partial t} = F_i(u_1,\dots,u_n,\nabla u_1,\dots,\nabla u_n,\nabla^2 u_1,\dots.\nabla^2 u_n,\dots), i=1,\dots,n;$

$u_i(\mathbf{x},0) = u_i^0(\mathbf{x}), \mathbf{x}\in\Omega$

$u_i(\mathbf{x},t) = g_i(\mathbf{x},t), \mathbf{x}\in\partial\Omega, t\in(0,T]$,

where $\nabla$ denotes spatial derivatives and $\partial\Omega$ denotes the boundary of the domain. For $0<k<n$, suppose measured data for solutions $\{u_1,\dots,u_k\}$ are available (\textit{data variables}) and the functions $\{F_{k+1},\dots,F_n\}$ (vector fields of the \textit{equation variables}) are known such that there is no overlap between the equation and data variables. 

**Goal:** Learn the full solution $\{u_1,\dots,u_k,\dots,u_n\}$ in a given time interval. The main model used is a neural network with multiple layers (https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)

### How to use this repository?
This repository contains four folders, one for each of the coupled systems in the paper. Each folder contains multiple files that include a python notebook containing the model training, a sample trained model that can be directly used to replicate some results in the paper, and any relevant datatset (if applicable).

If you want to run the training, please directly download the folder of your choice and run the respective .ipynb notebook. 

Note that the training and validation splits are random and so the results might vary slightly from the numbers reported in the paper.

The details of compute resources used in training of these models along with time taken for each of the models is given below. 

### Details of Computational Resources and Training Time
**Computation resources**: The authors used the Trillium Cluster under Digital Research Alliance of Canada (formerly known as Compute Canada). For some initial simulations, evaluations and plot generation, Google Colab was also used.

**Computational time:** 
#### SWE System

| Neurons \ Layers | 2    | 4    | 6    |
|------------------|-----:|-----:|-----:|
| 20               | 81.6 | 107  | 133.4|
| 50               | 82   | 108  | 136  |

The table above depicts the training time (in seconds) for the SWE system for different model complexities.

#### FN System
| Neurons \ Layers | 4    | 6    |
|------------------|-----:|-----:|
| 100              | 2104 | 6670 |
| 200              | 4576 | 15000|

The table above depicts the training time (in seconds) for the FN system for different model complexities.

#### RD System
| Neurons \ Layers | 4    | 6    |
|------------------|-----:|-----:|
| 50               | 3311 | 6092 |
| 100              | 6108 | 9300 |

The table above depicts the training time (in seconds) for the RD system for different model complexities.

#### Physical System of Wildfire
| Neurons \ Layers | 4    | 6    |
|------------------|-----:|-----:|
| 50               | 3073 | 3181 |
| 100              | 3089 | 3172 |

The table above depicts the training time (in seconds) for the physical system of wildfire for different model complexities. Note the trainng time doesn't differ much as we are using IHT and not the neuron $\ell_0$.

### Citation details

      @misc{title = {Learning Coupled System Dynamics under Incomplete Physical Constraints and Missing Data},
            url = {https://arxiv.org/pdf/2512.23761},
            author = {Saha, Esha and Wang, Hao},
            publisher = {arXiv},
            year = {2025}
    }




