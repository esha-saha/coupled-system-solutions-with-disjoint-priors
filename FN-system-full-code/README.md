We consider the Fitzhugh-Nagumo (FN) type reaction-diffusion system, in a 2D spatial domain $\Omega\subset \mathbb{R}^2$ with periodic boundary conditions, whose governing equations are two coupled PDEs described as

$u_t = \gamma_u \Delta u + u - u^3 - v + \alpha$

$v_t = \gamma_v \Delta v + \beta(u - v)$

where $u$ and $v$ represent two interactive components, $\gamma_u$ and $\gamma_v$ are diffusion coefficients, $\alpha$ and $\beta$ are the reaction coefficients, and $\Delta$ is the Laplacian operator. 
The FN equations are generally used to describe pattern formation in biological neuron activities excited by an external stimulus $\alpha$. The system exhibits an activator-inhibitor system where one equation boosts the production of both components while the other equation dissipates their new growth.
We generate the ground truth data by solving the system using a finite difference method with initial conditions taken as random vectors such that $u(x,y,0),v(x,y,0) \sim \mathcal{N}(\mathbf{0},0.2\mathbf{I}_2)$, where $\mathbf{0}$ and $\mathbf{I}_2$ denote the zero vector and $2\times 2$ identity matrix respectively.
The system is solved using $dx = dy = 0.5$ for $x,y \in [0,100]$ and $dt = 0.0005$ for $t\in [0,60]$. We select 50 evenly spaced data in time for $t\in [10,60]$ and 100  evenly spaced spatial data in both $x$ and $y$ direction to create the full-field ground truth dataset of size (100,100,50). 

Note: One of the the best trained neural network model that has been used to some of the figures in the paper have been given in fhn-5000Ns-lr-0.005-L0-0.0001-4Layers-200dim-Best. The simulated data Data_FHN_U.npy and Data_FHN_V.npy has been directly sourced from Chen at. al. (2021).
