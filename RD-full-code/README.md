## $\lambda-\omega$ reaction diffusion system

We consider a $\lambda-\omega$ reaction diffusion (RD) system in a 2D domain with the spiral pattern forming behavior governed by two coupled PDEs. Suppose $u$ and $v$ are two field variables, then the system of equations are given by

$u_t = 0.1 u_{xx} + 0.1 u_{yy} - u v^2 - u^3 + v^3 + u^2 v + u$ 

$v_t = 0.1 v_{xx} + 0.1 v_{yy} - u v^2 - u^3 - v^3 - u^2 v + v.$


The RD system can be used to describe wide range of behaviors including wave-like
phenomena and self-organized patterns found in chemical and biological systems. Some applications can be found in pattern formation, ecological invasions, etc. 
This particular RD equations in this test example displays spiral waves subject to periodic boundary conditions.
In order to generate the simulated training dataset, we solve the model using a finite difference method with inputs $x,y\in [-10,10]$ and $t\in [0,10]$. We downsample from a full solution with by uniformly selecting $128\times 128$ spatial points and 101 temporal steps.
Thus the dataset is of size $128\times128\times101$.

Note: 
