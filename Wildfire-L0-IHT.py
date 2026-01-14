import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
# import pandas as pd
# from pyDOE2 import lhs
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt
from scipy import stats
import time
import copy

def tridiagonal_matrix(m, n,diag_value, off_diag_above,off_diag_below):
    # Create a square matrix of size n x n filled with zeros
    matrix = np.zeros((m, n))

    # Set the main diagonal
    np.fill_diagonal(matrix, diag_value)

    # Set the first diagonal below the main diagonal
    np.fill_diagonal(matrix[1:], off_diag_below)

    # Set the first diagonal above the main diagonal
    np.fill_diagonal(matrix[:, 1:], off_diag_above)

    return matrix

def hpc(x,upc):
  if x>=upc:
    out = 1
  else:
    out = 0
  return out

def func_f(u,b,upc,e,a):
  f = np.zeros((u.shape[0],u.shape[1]))
  for i in range(u.shape[0]):
    for j in range(u.shape[1]):
      # print(np.linalg.norm(np.exp(u[i,j])),np.linalg.norm(1+e*u[i,j]))
      f[i,j] = hpc(u[i,j],upc)*b[i,j]*np.exp(u[i,j]/(1+e*u[i,j])) - a*u[i,j]
  # print(np.min(u))
  return f

def func_g(u,b,upc,e,q):
  g = np.zeros((u.shape[0],u.shape[1]))
  for i in range(u.shape[0]):
    for j in range(u.shape[1]):
      g[i,j] = -hpc(u[i,j],upc)*(e/q)*b[i,j]*np.exp(u[i,j]/(1+e*u[i,j])) #+ np.random.normal(0,0.1,1)[0]
  return g

#Define x-y-t spatial and temporal bounds
x_min = 0
x_max = 10
y_min = 0
y_max = 10
t_max = 10

#Define total number of discrete points
N_x = 50
N_y = 50
N_t = 100

#define step sizes
x_bounds = np.linspace(x_min,x_max,N_x+1)
y_bounds = np.linspace(y_min,y_max,N_y+1)
t_bounds = np.linspace(0,t_max,N_t)


#define finite difference matrices
D_x = tridiagonal_matrix(N_x+1,N_y+1, 0, -1,1)#to define d/dx
D_x[:3,0] = np.array([-3,4,-1])
D_x[N_x - 2:N_x+1,N_y] = np.array([1,-4,3])
D_y = D_x.T#to define d/dy

D_xx = tridiagonal_matrix(N_x+1,N_y+1, -2, 1,1)
D_xx[:4,0] = np.array([2,-5,4,-1])
D_xx[N_x - 3:N_x+1,N_y] = np.array([-1,4,-5,2])

D_yy = D_xx.T
U0 = np.loadtxt('/home/esaha/links/projects/def-wanghao/esaha/RD-Wildfire-system/U0.txt') #, sep=" ", header=None)
B0 = np.loadtxt('/home/esaha/links/projects/def-wanghao/esaha/RD-Wildfire-system/B0.txt')
# B0 = pd.read_csv('/home/esaha/links/projects/def-wanghao/esaha/RD-Wildfire-system/B0.txt', sep=" ", header=None)

B0 = np.ones((N_x+1,N_y+1))
print(x_bounds)
for i in range(N_x+1):
  for j in range(N_y+1):
    B0[i,j] = 0.1*(x_bounds[i]/10 + y_bounds[j]/10+5) #0.5*np.exp((x_bounds[i]/10 - 0.5)**2 + (y_bounds[j]/10 - 0.5)**2)
plt.imshow(B0)
plt.colorbar()

#Initialize Variables
u = np.zeros((N_x+1,N_y+1,N_t))
b = np.zeros((N_x+1,N_y+1,N_t))
v1 = np.zeros((N_x+1,N_y+1,N_t))
v2 = np.zeros((N_x+1,N_y+1,N_t))
F = np.zeros((N_x+1,N_y+1,N_t))
G = np.zeros((N_x+1,N_y+1,N_t))
V = np.zeros((N_t,2,N_x+1,N_y+1))

#Add in all initial values and parameters
u[:,:,0] = U0[:N_x+1,:N_x+1] #+ 1.5*np.roll(U0.values[:N_x+1,:N_x+1],-15,axis = 0) + np.roll(U0.values[:N_x+1,:N_x+1],-18,axis = 1)
b[:,:,0] = B0 #[:N_x+1,:N_x+1] #.values[:N_x+1,:N_x+1]
VV = np.load('/home/esaha/links/projects/def-wanghao/esaha/RD-Wildfire-system/VV.npy') #+ np.random.normal(0,1,(101,2,128,128)) #size is timesteps * 2 * N_x+1 * N_y+1
# V[:,0,:,:] = np.random.normal(0,0.5,(N_t,N_x+1,N_x+1))**2 #V[:N_t,:,:N_x+1,:N_y+1]
# V[:,1,:,:] = np.random.normal(0,0.5,(N_t,N_x+1,N_x+1))**2
V = VV[:N_t,:,:N_x+1,:N_y+1]
alpha = 1e-3
eps = 0.3
q = 1
kappa = 0.2
upc = 3


F[:,:,0] = func_f(u[:,:,0],b[:,:,0],upc,eps,alpha)
G[:,:,0] = func_g(u[:,:,0],b[:,:,0],upc,eps,q)

# plt.imshow(V[10,0,:,:],origin='lower')
# plt.colorbar()

# for i in range(1,N_t):
#   # print(np.linalg.norm(np.exp(u[:,:,i])))
#   F[1:-1,1:-1,i] = func_f(u[1:-1,1:-1,i-1],b[1:-1,1:-1,i-1],upc,eps,alpha)
#   G[1:-1,1:-1,i] = func_g(u[1:-1,1:-1,i-1],b[1:-1,1:-1,i-1],upc,eps,q)
#   u[1:-1,1:-1,i] = u[1:-1,1:-1,i-1] + (t_bounds[1]-t_bounds[0])*(kappa*(np.matmul(u[1:-1,1:-1,i-1],D_xx[1:-1,1:-1]) +
#                                   np.matmul(D_yy[1:-1,1:-1],u[1:-1,1:-1,i-1])) -
#                             (V[i-1,0,1:N_x,1:N_x]*(np.matmul(u[1:-1,1:-1,i-1],D_x[1:-1,1:-1])) + V[i-1,1,1:N_y,1:N_y]*(np.matmul(D_y[1:-1,1:-1],u[1:-1,1:-1,i-1]))) +
#                             func_f(u[1:-1,1:-1,i-1],b[1:-1,1:-1,i-1],upc,eps,alpha))
#   b[1:-1,1:-1,i] = b[1:-1,1:-1,i-1] + (t_bounds[1]-t_bounds[0])*func_g(u[1:-1,1:-1,i-1],b[1:-1,1:-1,i-1],upc,eps,q)
#   if i%10==0:
#     print('time step is',i)

for i in range(1,N_t):
  # print(np.linalg.norm(np.exp(u[:,:,i])))
  F[:,:,i] = func_f(u[:,:,i-1],b[:,:,i-1],upc,eps,alpha)
  G[:,:,i] = func_g(u[:,:,i-1],b[:,:,i-1],upc,eps,q)
  u[:,:,i] = u[:,:,i-1] + (t_bounds[1]-t_bounds[0])*(kappa*(np.matmul(u[:,:,i-1],D_xx) +
                                  np.matmul(D_yy,u[:,:,i-1])) -
                            (V[i-1,0,:N_x+1,:N_x+1]*(np.matmul(u[:,:,i-1],D_x)) + V[i-1,1,:N_y+1,:N_y+1]*(np.matmul(D_y,u[:,:,i-1]))) +
                            func_f(u[:,:,i-1],b[:,:,i-1],upc,eps,alpha))
  b[:,:,i] = b[:,:,i-1] + (t_bounds[1]-t_bounds[0])*func_g(u[:,:,i-1],b[:,:,i-1],upc,eps,q)
  if i%10==0:
    print('time step is',i)


noise_std = [20] #,1000,5000] #,5000]
for noise in noise_std:
  print(noise,'Nt================6 LAYERS 50 DIM 1000 NST==================')
  t_bounds = np.linspace(0,1,80)
  x_bounds = np.linspace(0,1,N_x + 1)
  y_bounds = np.linspace(0,1,N_y + 1)
  data_u_norm = np.zeros((u.shape[0],u.shape[1],N_t))
  data_b_norm = np.zeros((b.shape[0],b.shape[1],N_t))
  data_v1_norm = np.zeros((N_t,V.shape[2],V.shape[3]))
  data_v2_norm = np.zeros((N_t,V.shape[2],V.shape[3]))
  u_min_vec = np.zeros(N_t)
  u_max_vec = np.zeros(N_t)
  b_min_vec = np.zeros(N_t)
  b_max_vec = np.zeros(N_t)
  v1_min_vec = np.zeros(N_t)
  v1_max_vec = np.zeros(N_t)
  v2_min_vec = np.zeros(N_t)
  v2_max_vec = np.zeros(N_t)

  for i in range(N_t):
    u_min = np.min(u[:,:,i])
    u_max = np.max(u[:,:,i])
    data_u_norm[:,:,i] = (u[:,:,i] - u_min)/(u_max - u_min)

    b_min = np.min(b[:,:,i])
    b_max = np.max(b[:,:,i])
    data_b_norm[:,:,i] = (b[:,:,i] - b_min)/(b_max - b_min)

    v1_min = np.min(V[i,0,:,:])
    v1_max = np.max(V[i,0,:,:])
    if v1_min == v1_max:
      data_v1_norm[i,:,:] = (V[i,0,:,:] - v1_min)
    else:
      data_v1_norm[i,:,:] = (V[i,0,:,:] - v1_min)/(v1_max - v1_min)
    # print(V[i,0,0,:])

    v2_min = np.min(V[i,1,:,:])
    v2_max = np.max(V[i,1,:,:])
    if v2_min==v2_max:
      data_v2_norm[i,:,:] = (V[i,1,:,:] - v2_min)
    else:
      data_v2_norm[i,:,:] = (V[i,1,:,:] - v2_min)/(v2_max - v2_min)

    u_min_vec[i] = u_min
    u_max_vec[i] = u_max
    b_min_vec[i] = b_min
    b_max_vec[i] = b_max

  data_u = data_u_norm
  data_b = data_b_norm
  data_v1 = data_v1_norm
  data_v2 = data_v2_norm
  plt.plot(data_b_norm.reshape(-1))
  plt.imshow(data_b[:,:,30])
  plt.colorbar()

  # print(u.shape)
  t1 = 20
  t2 = 100
  nt = int(t2-t1)
  u_data = data_u[:,:,t1:t2].reshape((N_x+1)*(N_y+1),nt)
  b_data = data_b[:,:,t1:t2].reshape((N_x+1)*(N_y+1),nt)
  # print('original u',u.shape,u[:,:,0],'\nu.reshape((N_x+1)*(N_y+1),N_t)',u_data.shape,u_data)
  u0_data = data_u[:,:,t1].reshape((N_x+1)*(N_y+1),1)
  b0_data = data_b[:,:,t1].reshape((N_x+1)*(N_y+1),1)
  # v1_data = V[t1:t2,0,:,:].reshape((N_x+1)*(N_y+1),nt)
  # v2_data = V[t1:t2,1,:,:].reshape((N_x+1)*(N_y+1),nt)
  v1_data = data_v1[t1:t2,:,:].reshape((N_x+1)*(N_y+1),nt)
  v2_data = data_v2[t1:t2,:,:].reshape((N_x+1)*(N_y+1),nt)
  F_data = F[:,:,t1:t2].reshape((N_x+1)*(N_y+1),nt)
  G_data = G[:,:,t1:t2].reshape((N_x+1)*(N_y+1),nt)

  t_data = t_bounds #[t1:t2]
  # print(t_bounds)
  t_data = np.tile(t_data,((N_x+1)*(N_y+1),1))


  x_data = x_bounds.reshape(-1,1)
  x_data = np.tile(x_data, (1, N_x+1))
  x_data = np.reshape(x_data, (-1, 1))
  x_data = np.tile(x_data, (1, nt))

  y_data = y_bounds.reshape((1,-1)) #Note this reshape is (1,-1) and NOT (-1,1)
  y_data = np.tile(y_data, ((N_y+1), 1))
  y_data = np.reshape(y_data, (-1, 1))
  y_data = np.tile(y_data, (1, nt))

  # print(x_data,'\n',y_data)
  N_s = 1000
  steps = 60
  print('N_s and N_t are:',N_s,N_t)
  idx_s = np.random.choice(x_data.shape[0], N_s, replace = False)
  # idx_y = np.random.choice(x_bounds.shape[0], N_s, replace = False)
  idx_t = np.random.choice(nt,steps, replace = False)

  u_max = np.tile(u_max_vec[t1:t2],(((N_x+1)*(N_y+1)),1)).squeeze().reshape((N_x+1)*(N_y+1),nt)
  u_min = np.tile(u_min_vec[t1:t2],(((N_x+1)*(N_y+1)),1)).squeeze().reshape((N_x+1)*(N_y+1),nt)
  b_max = np.tile(b_max_vec[t1:t2],(((N_x+1)*(N_y+1)),1)).squeeze().reshape((N_x+1)*(N_y+1),nt)
  b_min = np.tile(b_min_vec[t1:t2],(((N_x+1)*(N_y+1)),1)).squeeze().reshape((N_x+1)*(N_y+1),nt)
  # print(idx_t)

  t_meas = t_data[idx_s, :]
  t_meas = t_meas[:, idx_t].reshape((-1,1))
  x_meas = x_data[idx_s, :]
  x_meas = x_meas[:, idx_t].reshape((-1,1))
  y_meas = y_data[idx_s, :]
  y_meas = y_meas[:, idx_t].reshape((-1,1))
  u_meas = u_data[idx_s, :]
  u_meas = u_meas[:, idx_t].reshape((-1,1))
  b_meas = b_data[idx_s, :]
  b_meas = b_meas[:, idx_t].reshape((-1,1))
  u_max_meas = u_max[idx_s,:][:,idx_t].reshape((-1,1))
  u_min_meas = u_min[idx_s,:][:,idx_t].reshape((-1,1))
  b_max_meas = b_max[idx_s,:][:,idx_t].reshape((-1,1))
  b_min_meas = b_min[idx_s,:][:,idx_t].reshape((-1,1))

  v1_meas = v1_data[idx_s, :]
  v1_meas = v1_meas[:, idx_t].reshape((-1,1))
  v2_meas = v2_data[idx_s, :]
  v2_meas = v2_meas[:, idx_t].reshape((-1,1))
  F_meas = F_data[idx_s, :]
  F_meas = F_meas[:, idx_t].reshape((-1,1))
  G_meas = G_data[idx_s, :]
  G_meas = G_meas[:, idx_t].reshape((-1,1))

  X_meas = np.hstack((x_meas, y_meas, t_meas))
  # print(X_meas,'\n','\n',u_meas.T)
  # print(x_bounds,'\n',u[:,:,0])

  Split_TrainVal = 0.8
  N_train = int(N_s*steps*Split_TrainVal)
  idx_train = np.random.choice(X_meas.shape[0], N_train, replace=False)
  X_train = X_meas[idx_train,:]
  u_train = u_meas[idx_train,:]
  b_train = b_meas[idx_train,:]
  v1_train = v1_meas[idx_train,:]
  v2_train = v2_meas[idx_train,:]
  F_train = F_meas[idx_train,:]
  G_train = G_meas[idx_train,:]
  u_max_train = u_max_meas[idx_train,:]
  u_min_train = u_min_meas[idx_train,:]
  b_max_train = b_max_meas[idx_train,:]
  b_min_train = b_min_meas[idx_train,:]
  # Validation Measurements, which are the rest of measurements
  idx_val = np.setdiff1d(np.arange(X_meas.shape[0]), idx_train, assume_unique=True)
  X_val = X_meas[idx_val,:]
  u_val = u_meas[idx_val,:]
  b_val = b_meas[idx_val,:]
  v1_val = v1_meas[idx_val,:]
  v2_val = v2_meas[idx_val,:]
  F_val = v2_meas[idx_val,:]
  u_max_val = u_max_meas[idx_val,:]
  u_min_val = u_min_meas[idx_val,:]
  b_max_val = b_max_meas[idx_val,:]
  b_min_val = b_min_meas[idx_val,:]

  print(t1,t2)


  def m(x):
    return torch.relu(x)
  
  def compute_residuals(u, b, x, y, t, umin, umax, bmin, bmax, v1,v2):
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True,allow_unused=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x , create_graph=True,allow_unused=True)[0]
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True,allow_unused=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y , create_graph=True,allow_unused=True)[0]
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True,allow_unused=True)[0]
    b_t = torch.autograd.grad(b.sum(), t, create_graph=True,allow_unused=True)[0]
    c1 = (umax - umin) #.reshape(-1)
    c2 = (bmax - bmin)#.reshape(-1)
    # print(c1.shape,c2.shape,u_t.shape,u.shape)
    f = Variable(torch.from_numpy(func_f((c1*u + umin).detach().cpu().numpy().reshape(-1,1),
                                         (c2*b + bmin).detach().cpu().numpy().reshape(-1,1),upc,eps,alpha)).float()
                                         , requires_grad=True).to(device)
    g = Variable(torch.from_numpy(func_g((c1*u + umin).detach().cpu().numpy().reshape(-1,1),
                                         (c2*b + bmin).detach().cpu().numpy().reshape(-1,1),upc,eps,q)).float()
                                         , requires_grad=True).to(device)

    pde_u = -(c1/10)*u_t + kappa*(c1**2/100)*(u_yy + u_xx) - v1.reshape(-1,1)*(c1/10)*u_x - v2.reshape(-1,1)*(c1/10)*u_y  + (-g/0.3 - 0.001*c1*(u+umin).reshape(-1,1)) #(0.1*u_xx + 0.1*u_yy - u*v**2 - u**3 + v**3 +u**2*v + u - u_t).reshape(-1,1)
    # pde_u = (0.1*(c_u/L**2)*u_xx + 0.1*(c_u/L**2)*u_yy - (c_u*u+umin)*(c_v*v+vmin)**2 - (c_u*u+umin)**3 + (c_v*v+vmin)**3 + (c_u*u+umin)**2*(c_v*v+vmin) + (c_u*u+umin) - (c_u/10)*u_t).reshape(-1,1)
    pde_b = (- (c2/(8))*(b_t.reshape(-1,1)) + g.reshape(-1,1)).reshape(-1,1)
    # print(c1.shape,c2.shape,u_t.shape,u.shape,pde_u.shape)
    return pde_u,pde_b

     
  def hard_threshold(model, sparsity):
    """
    Zero-out smallest weights BUT keep their gradients working.
    """
    all_weights = []
    for param in model.parameters():
        if param.requires_grad:
            all_weights.append(param.data.view(-1))

    # concatenate all
    all_weights = torch.cat(all_weights)

    # find threshold
    k = int((1 - sparsity) * all_weights.numel())  # number to keep
    if k < 1:
        return

    threshold = torch.topk(all_weights.abs(), k, largest=True).values.min()

    # apply threshold
    for param in model.parameters():
        mask = (param.data.abs() >= threshold).float()
        param.data *= mask
  
  def nnz_percentage(model):
    nz = 0
    total = 0
    for p in model.parameters():
        total += p.numel()
        nz += (p.data != 0).sum().item()
    return 100 * nz / total
  
  def cart_inputs(x,y,t):
    a = np.array([[x0, y0,t0] for x0 in x for y0 in y for t0 in t])
    return a[:,0].reshape(-1,1), a[:,1].reshape(-1,1), a[:,2].reshape(-1,1)

  class Net(nn.Module):
    def __init__(self,H):
        super(Net, self).__init__()
        self.hidden_layer1 = nn.Linear(3,H)
        self.hidden_layer2 = nn.Linear(H,H)
        self.hidden_layer3 = nn.Linear(H,H)
        self.hidden_layer4 = nn.Linear(H,H)
        # self.hidden_layer5 = nn.Linear(H,H)
        # self.hidden_layer6 = nn.Linear(H,H)
        self.output_layer = nn.Linear(H,2)
        
    def forward(self, x,y,t,b_in,v1,v2,umin,umax,bmin,bmax):
        # inputs = torch.cat([x,y,t,b_in,v1.reshape(-1,1),v2.reshape(-1,1)],axis=1) # combined two arrays of 1 columns each to one array of 3 columns
        # print(inputs)
        inputs = torch.cat([x,y,t],axis=1)
        layer1_out = m(self.hidden_layer1(inputs))
        layer2_out = m(self.hidden_layer2(layer1_out))
        layer3_out = m(self.hidden_layer3(layer2_out))
        layer4_out = m(self.hidden_layer4(layer3_out))
        # layer5_out = m(self.hidden_layer5(layer4_out))
        # layer6_out = m(self.hidden_layer6(layer5_out))
        output = self.output_layer(layer4_out)
        # outputb = self.output_layerb(b1)
        u = output[:,0].reshape(-1,1)
        b = output[:,1].reshape(-1,1)

        pde_u = 0 # -u_t + kappa*(u_yy + u_xx) - v1.reshape(-1,1)*u_x - v2.reshape(-1,1)*u_y  + (-g/0.3 - 0.001*u.reshape(-1,1))
        # print(b_t.shape,g.shape,c2.shape,u.shape,b.shape)
        pde_b = 0 #(- (c2/(100))*(b_t.reshape(-1)) + g.reshape(-1)).reshape(-1,1) # 0.3*(f + 0.001*u.reshape(-1,1))
        # print('pde_b',pde_b.shape)
        return u,b,pde_u,pde_b


  x_ic,y_ic,t_ic = cart_inputs(x_bounds,y_bounds,t_bounds[0]*np.ones((1)))
  u_ic = torch.tensor(data_u[:N_x+1,:N_x+1,t1]).reshape(-1).reshape(-1,1).float() #.detach().numpy() #U0.values[:N_x+1,:N_x+1].reshape(-1) #.repeat(N_t,1)
  b_ic = torch.tensor(data_b[:N_x+1,:N_x+1,t1]).reshape(-1).reshape(-1,1).float() #.detach().numpy() #B0.values[:N_x+1,:N_x+1].reshape(-1)

  pt_x_ic = Variable(torch.from_numpy(x_ic).float(), requires_grad=True).to(device)
  pt_y_ic = Variable(torch.from_numpy(y_ic).float(), requires_grad=True).to(device)
  pt_t_ic = Variable(torch.from_numpy(t_ic).float(), requires_grad=True).to(device)

  pt_u_ic = Variable(u_ic, requires_grad=True).to(device)
  pt_b_ic = Variable(b_ic, requires_grad=True).to(device)

  pt_V = Variable(torch.from_numpy(V[t1:t2,:,:,:]).float(), requires_grad=True).to(device)

  x_collocation = X_train[:,0].reshape(-1,1) #np.random.uniform(low=x_min, high=x_max, size=(N_x+1,1))
  y_collocation = X_train[:,1].reshape(-1,1) #np.random.uniform(low=y_min, high=y_max, size=(N_y+1,1))
  t_collocation = X_train[:,2].reshape(-1,1)

  all_zeros = np.zeros((X_train.shape[0],1))


  pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True).to(device)
  pt_y_collocation = Variable(torch.from_numpy(y_collocation).float(), requires_grad=True).to(device)
  pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True).to(device)
  v1_col = Variable(torch.from_numpy(v1_train).float(), requires_grad=True).to(device)
  v2_col = Variable(torch.from_numpy(v2_train).float(), requires_grad=True).to(device)
  pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)

  x_val = X_val[:,0].reshape(-1,1) #np.random.uniform(low=x_min, high=x_max, size=(N_x+1,1))
  y_val = X_val[:,1].reshape(-1,1) #np.random.uniform(low=y_min, high=y_max, size=(N_y+1,1))
  t_val = X_val[:,2].reshape(-1,1)

  pt_x_val = Variable(torch.from_numpy(x_val).float(), requires_grad=True).to(device)
  pt_y_val = Variable(torch.from_numpy(y_val).float(), requires_grad=True).to(device)
  pt_t_val = Variable(torch.from_numpy(t_val).float(), requires_grad=True).to(device)


  # === Hyperparameters ===
  learning_rates = [5e-3] #,1e-6,1e-5] #,1e-4]
  lr=5e-3
  hidden_dims = [50,100]
  # hidden_dim = 50
  lam_0 = [0.25,0.75]
  num_repeats = 3
  validate_every = 2000
  max_epochs = 20000
  patience = 5

  best_global_val_loss = float('inf')
  best_model_state = None
  best_hparams = {}
  mse_cost_function1 = torch.nn.MSELoss() # Mean squared error
  results = []

  # for lr in learning_rates:
  for hidden_dim in hidden_dims:
      for lambda0 in lam_0:
          run_errors = []

          print(f"\n=== Training with lr={lr}, hidden_dim={hidden_dim} ===sparsity (zeros) = {lambda0}")

          for run in range(num_repeats):
              print(f"Run {run + 1}/{num_repeats}")
              torch.manual_seed(run)
              

              net = Net(hidden_dim).to(device)
              optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay = 0.0001)

              best_val_loss = float('inf')
              best_model_wts = copy.deepcopy(net.state_dict())
              epochs_no_improve = 0
              t_start_train = time.time()

              for epoch in range(0, max_epochs):
                  net.train()
                  optimizer.zero_grad()

                  uout, bout, _, _ = net(pt_x_collocation,pt_y_collocation,pt_t_collocation,torch.tensor(b_train).float().to(device),
                                  torch.tensor(v1_train).float().to(device),torch.tensor(v2_train).float().to(device),torch.tensor(u_min_train).float().to(device),
                                  torch.tensor(u_max_train).float().to(device),torch.tensor(b_min_train).float().to(device),
                                  torch.tensor(b_max_train).float().to(device))
                  
                  f_out, g_out = compute_residuals(
                      uout, bout,
                      pt_x_collocation, pt_y_collocation, pt_t_collocation,
                      torch.tensor(u_min_train).float().to(device),
                      torch.tensor(u_max_train).float().to(device),
                      torch.tensor(b_min_train).float().to(device),
                      torch.tensor(b_max_train).float().to(device),
                      torch.tensor(v1_train).float().to(device),torch.tensor(v2_train).float().to(device)
                  )

                  net_uic, net_bic, _, _ = net(pt_x_ic, pt_y_ic, pt_t_ic,pt_b_ic,
                              pt_V[t1,0,:,:].reshape(-1),pt_V[t1,1,:,:].reshape(-1),torch.tensor(u_min[:,0]).float().to(device),torch.tensor(u_max[:,0]).float().to(device),
                              torch.tensor(b_min[:,0]).float().to(device),torch.tensor(b_max[:,0]).float().to(device))

                  u_out = uout.reshape(-1, 1)
                  b_out = bout.reshape(-1, 1)
                  # print(g_out.shape,pt_all_zeros.shape,net_bic.shape,pt_b_ic.shape,u_out.shape,b_train.shape)

                  mse_u = mse_cost_function1(f_out, pt_all_zeros)
                  mse_b = mse_cost_function1(g_out, pt_all_zeros)
                  mse_uic = mse_cost_function1(net_uic.reshape(-1, 1), pt_u_ic)
                  mse_bic = mse_cost_function1(net_bic.reshape(-1, 1), pt_b_ic) #torch.from_numpy(u_ic.reshape(-1, 1)).float().to(device))
                  mse_udata = mse_cost_function1(u_out, torch.tensor(u_train).float().to(device))
                  mse_bdata = mse_cost_function1(b_out, torch.tensor(b_train).float().to(device))

                  # Weight the regularization
                  # lambda_l0 = 1e-5  # tune this

                  loss = mse_udata + mse_b + mse_bic 
                  # loss = mse_bdata + mse_u + mse_uic
                  loss.backward()
                  optimizer.step()

                  if epoch % validate_every == 0 and epoch > 10000:
                      hard_threshold(net, sparsity=lambda0)
                      net.eval()
                      # with torch.no_grad():
                      if epoch%2000==0:
                        u_outval,b_outval,_,_ = net(pt_x_val,pt_y_val,pt_t_val,torch.tensor(b_val).float().to(device),
                                                    torch.tensor(v1_val).float().to(device),torch.tensor(v2_val).float().to(device),torch.tensor(u_min_val).float().to(device),
                                                    torch.tensor(u_max_val).float().to(device),torch.tensor(b_min_val).float().to(device),
                                                    torch.tensor(b_max_val).float().to(device))
                        
                      #   full_uv_val = torch.hstack((torch.tensor(u_val),torch.tensor(v_val)))
                      #   net_uv_val = torch.hstack((u_outval,v_outval)).detach()
                      #   rel_uv = torch.norm(torch.tensor(full_uv_val).reshape(-1).to(device) - net_uv_val.reshape(-1)) / \
                      #           torch.norm(torch.tensor(full_uv_val).reshape(-1).to(device))

                        rel_u = torch.norm(torch.tensor(u_val).reshape(-1).to(device) - u_outval.reshape(-1)) / \
                                torch.norm(torch.tensor(u_val).reshape(-1).to(device))

                        rel_b = torch.norm(torch.tensor(b_val).reshape(-1).to(device) - b_outval.reshape(-1)) / \
                                torch.norm(torch.tensor(b_val).reshape(-1).to(device))

                        val_loss = 0.5 * (rel_u + rel_b)
                        print('At epoch',epoch,':',rel_u.item(),rel_b.item())

                        if val_loss.item() < best_val_loss - 1e-6:
                            best_val_loss = val_loss.item()
                            best_model_wts = copy.deepcopy(net.state_dict())
                            epochs_no_improve = 0
                        else:
                            epochs_no_improve += 1

                        if epochs_no_improve >= patience:
                            print(f"Early stopping at epoch {epoch}")
                            break

              t_end_train = time.time()
              # # Load best weights
              net.load_state_dict(best_model_wts)

              # Evaluate final validation score
              s=1
              xx= x_bounds[::s] #np.linspace(x_min,x_max,41)
              yy= y_bounds[::s] #np.linspace(x_min,x_max,41)
              tt= t_data[0,:] #np.linspace(0,10,100)
              x1,y1,tt1 = cart_inputs(xx,yy,tt)
              pt_x = Variable(torch.from_numpy(x1).float(), requires_grad=True).to(device)
              pt_y = Variable(torch.from_numpy(y1).float(), requires_grad=True).to(device)
              pt_t = Variable(torch.from_numpy(tt1).float(), requires_grad=True).to(device)
              pt_u,pt_b,_,_ = net(pt_x,pt_y,pt_t,torch.tensor(b_data).reshape(-1,1).float().to(device),
                                  torch.tensor(v1_data).reshape(-1,1).to(device).float(),
                                  torch.tensor(v2_data).reshape(-1,1).float().to(device),
                                  torch.tensor(u_max[::s,:]).reshape(-1,1).float().to(device),torch.tensor(u_min[::s,:]).reshape(-1,1).float().to(device),
                                  torch.tensor(b_max[::s,:]).reshape(-1,1).float().to(device),torch.tensor(b_min[::s,:]).reshape(-1,1).float().to(device))
              ms_u = pt_u.reshape(xx.shape[0],yy.shape[0],tt.shape[0])
              ms_b = pt_b.reshape(xx.shape[0],yy.shape[0],tt.shape[0])
              u_true = data_u[:,:,t1:t2]
              b_true = data_b[:,:,t1:t2]

              full_field_true = torch.hstack((torch.tensor(u_true),torch.tensor(b_true))).to(device)
              full_field_net = torch.hstack((ms_u,ms_b)).detach()

              error_uv = torch.zeros(nt)
              error_u = torch.zeros(nt)
              error_b = torch.zeros(nt)
              for i in range(nt):
                error_uv[i] = torch.norm(full_field_true[:,:,i] - full_field_net[:,:,i])/torch.norm(full_field_true[:,:,i])
                error_u[i] = torch.norm(torch.tensor(u_true[:,:,i]).to(device) - ms_u[:,:,i])/torch.norm(torch.tensor(u_true[:,:,i]).to(device))
                error_b[i] = torch.norm(torch.tensor(b_true[:,:,i]).to(device) - ms_b[:,:,i])/torch.norm(torch.tensor(b_true[:,:,i]).to(device))


            #   error_u = torch.zeros(N_t)
            #   error_v = torch.zeros(N_t)
            #   for i in range(N_t):
            #     error_u[i] = torch.norm(torch.tensor(data_u[::s,::s,i]).to(device) - ms_u[:,:,i].detach())/torch.norm(torch.tensor(data_u[::s,::s,i]).to(device))
            #     error_v[i] = torch.norm(torch.tensor(data_v[::s,::s,i]).to(device) - ms_v[:,:,i].detach())/torch.norm(torch.tensor(data_v[::s,::s,i]).to(device))

              print('\nError uv',torch.mean(error_uv),' Error u',torch.mean(error_u),'Error b:', torch.mean(error_b),'\n')

              final_error = torch.mean(error_uv) #+ torch.mean(error_v))
              run_errors.append(final_error.item())
              print("\nNNZ% =", nnz_percentage(net))
              # total_w, nnz_w = count_weights_and_nnz(net)
              # perc = 100 * nnz_w / total_w

              print(f"Run {run+1}/{num_repeats} Final Val Error: {final_error.item():.6f}")
              # print(f"\nTotal weights: {total_w}")
              # print(f"Non-zero (active) weights: {nnz_w}")
              # print(f"Percentage active: {perc:.2f}%")
              print(f"Total training time is",t_end_train-t_start_train)

          # Compute mean ± 95% CI
          mean_error = np.mean(run_errors)
          sem = stats.sem(run_errors)
          ci95 = sem * stats.t.ppf((1 + 0.95) / 2., num_repeats - 1)

          print(f"At lr={lr}, hidden_dim={hidden_dim} mean full-field L2 Error: {mean_error:.6f} ± {ci95:.6f}")
          
          torch.save(net,f'/home/esaha/links/scratch/L0-trained-models-outputs/Wildfire-L0-models/wildfire-L0-FixedWind-Ns-{N_s}-{lambda0}PercSparsity')
          # results.append((lr, hidden_dim, mean_error, ci95))

          # if mean_error < best_global_val_loss:
          #     best_global_val_loss = mean_error
          #     best_model_state = copy.deepcopy(net.state_dict())
          #     best_hparams = {'lr': lr, 'hidden_dim': hidden_dim}

  # Final results
#   for r in results:
#       print(f"lr={r[0]:.0e}, hidden_dim={r[1]:>3} → Val Error: {r[2]:.6f} ± {r[3]:.6f}")

  print(f"Best Hyperparameters: {best_hparams}, Validation Error: {best_global_val_loss:.6f}")
  # torch.save(net,f'/home/esaha/links/scratch/RD-results/rd-wildfire-test-best-L0-{lambda0}-lr-{lr}-Ns-{N_s}-{hidden_dim}dim')