## Physical Model of Wildfires

Physical models of wildfire are built from conservation laws which model the rate of change of fire temperature based on conservation of energy, balance of fuel supply and fuel reaction rate. The coupled system is reaction-diffusion like system, has been described in multiple works \citep{mandel2008wildland,san20232d,8966412} and is given as,

$u_t = \kappa\Delta u - \mathbf{v}\cdot\nabla u + f(u,\beta), text{ in } \Omega \times (0,t_{max}]$

$\beta_t  = g(u,\beta), \text{ in } \Omega \times (0,t_{max}]$

where $u$ and $\beta$ denote the firefront temperature and fuel availability respectively, $t_{max}$ denotes the final time, $\mathbf{v}$ denotes the vector field of wind and topography. The the definition of other constants and the values used are given in Table \ref{tab:symbols}. The functions $f$ and $g$ can be described in various ways. 

$f(u,\beta) = H_{pc}(u)\beta\exp\left(\dfrac{u}{1+\epsilon u}\right) - \alpha u$

$g(u,\beta) = -H_{pc}(u)\dfrac{\epsilon}{q}\beta\exp\left(\dfrac{u}{1+\epsilon u}\right)$

where $H_{pc}(u) = \[1, \text{if } u\geq u_{pc} \text{ and } 0, \text{ otherwise} \]$
 
Assuming that the spatial domain is large enough to avoid fire spreading at the boundary $\partial\Omega$, we assume Dirichlet boundary conditions $u_{\partial\Omega}(\mathbf{x},t) =\beta_{\partial\Omega}(\mathbf{x},t) = 0$ on $\partial\Omega\times (0,t_{max}].$
This model can be solved numerically to obtain fire spread dynamics in a given spatio-temporal domain.

 | Symbol      | Description                                   | Values                  |
|-------------|-----------------------------------------------|-------------------------|
| $\Omega$    | Spatial domain                                | $[0,10]\times[0,10]$    |
| $t_{max}$   | Final time                                    | 10                      |
| $\kappa$    | Diffusion coefficient                         | 0.2                     |
| $\epsilon$  | Inverse of activation energy of fuel          | 0.3                     |
| $\alpha$    | Natural convection                            | 0.01                    |
| $q$         | Reaction heat                                 | 1                       |
| $u_{pc}$    | Phase change threshold                        | 3                       |

Notes:
1. One of the the best trained neural network model that has been used to some of the figures in the paper have been given in rd-wildfire-test-L0-RandomWind-lr-0.005-Ns-1000-100dim.

2. The file VV.npy is too large to upload on github. Please use unzip VV-npy.zip for the relevant file.
