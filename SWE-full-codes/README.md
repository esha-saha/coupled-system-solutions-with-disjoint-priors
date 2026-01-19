This folder contains implementations of the SWE system of equations.

Given $x\in \Omega$, where $\Omega\subset\mathbb{R}$ denotes the (spatial) input domain, suppose $h$ denotes the height of water, and $u$ denotes the horizontal velocity. For $g = 9.81$ $ms^{-2}$ (gravitational constant) and bed surface $z_b$, the (rearranged) 1D SWE is given by

$h_t + hu_x = 0$ 

$(hu)_t + \left(hu^2 + \frac{1}{2}gh^2\right)_x + g(z_b)_x=0$.

 
We apply this system to the well-known dam break problem. For $\Omega\times [0,1]$, where $\Omega = [0,10]$, assuming bed surface $z_b = 0$, we obtain the full-field solution of this system using the initial conditions, 

$h(x,0) =  \left( 1 \text{ if } x\in [0,5] \text{ and } 0 \text{ if } x\in (5,1] \right)$

$u(x,0) = 0$.
   
The boundary conditions are chosen such that there is no flow in or out at the boundaries i.e., a zero-gradient (Neumann) boundary conditions with $\dfrac{\partial h}{\partial x} = \dfrac{\partial u}{\partial x} = 0$ for $x\in\partial\Omega$, where $\partial\Omega$ denotes the boundary. The Local Lax-Friedrichs (Rusanov flux) finite volume method is used to obtain the solutions with $\Delta x = 0.5$ and $\Delta t = 0.001$. 

Note: One of the the best 4 layer trained neural network model that has been used to some of the figures in the paper have been given in swe_L0_4L_best_dim-20-lr-0.005_L0-1e-06.
