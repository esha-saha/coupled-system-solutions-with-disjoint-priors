import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
# from pyDOE2 import lhs
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt
from scipy import stats
import time
import copy

noise_std = [0.0]
for noise in noise_std:
  print('N_T',noise,'===============THIS IS WITH 4 LAYERS, 100 DIM, 5000 NS, 40NT =================')
  # Parameters
  L = 100.0
  N = 200
  dx = L / N
  dt = 0.0005
  steps = 120000
  print('time interval is [0,',dt*steps,']')

  gamma_u = 1
  gamma_v = 100
  alpha = 0.01
  beta = 0.25

  # Create grid
  # x = np.linspace(0, L, N, endpoint=False)
  # y = np.linspace(0, L, N, endpoint=False)
  # X, Y = np.meshgrid(x, y)

  data_u_clean = np.load('/home/esaha/links/projects/def-wanghao/esaha/fhn_codes_data/Data_FHN_U.npy').T[::2,::2,:] 
  data_v_clean = np.load('/home/esaha/links/projects/def-wanghao/esaha/fhn_codes_data/Data_FHN_V.npy').T[::2,::2,:] 
  N_x = data_u_clean.shape[0]-1
  N_y = N_x
  N_t = 50
  data_u_norm_clean = np.zeros((data_u_clean.shape[0],data_u_clean.shape[1],N_t))
  data_v_norm_clean = np.zeros((data_v_clean.shape[0],data_v_clean.shape[1],N_t))
  for i in range(N_t):
    u_min_clean = np.min(data_u_clean[:,:,i])
    u_max_clean = np.max(data_u_clean[:,:,i])
    data_u_norm_clean[:,:,i] = (data_u_clean[:,:,i] - u_min_clean)/(u_max_clean - u_min_clean)

    v_min_clean = np.min(data_v_clean[:,:,i])
    v_max_clean = np.max(data_v_clean[:,:,i])
    data_v_norm_clean[:,:,i] = (data_v_clean[:,:,i] - v_min_clean)/(v_max_clean - v_min_clean)

  
  data_u = data_u_clean #+ np.random.normal(0,noise,(100,100,50))
  data_v = data_v_clean #+ np.random.normal(0,noise,(100,100,50))
  print(data_u.shape)
  plt.imshow(data_u[:,:,5])
  # print('time interval is [0,',dt*steps,']')
  print('Selected time interval is [10,',0.0005*steps,']')



  t_bounds = np.linspace(0,1,N_t)
  x_bounds = np.linspace(0,1,N_x + 1)
  y_bounds = np.linspace(0,1,N_y + 1)
  data_u_norm = np.zeros((data_u.shape[0],data_u.shape[1],N_t))
  data_v_norm = np.zeros((data_v.shape[0],data_v.shape[1],N_t))
  u_min_vec = np.zeros(N_t)
  u_max_vec = np.zeros(N_t)
  v_min_vec = np.zeros(N_t)
  v_max_vec = np.zeros(N_t)

  for i in range(N_t):
    u_min = np.min(data_u[:,:,i])
    u_max = np.max(data_u[:,:,i])
    data_u_norm[:,:,i] = (data_u[:,:,i] - u_min)/(u_max - u_min)

    v_min = np.min(data_v[:,:,i])
    v_max = np.max(data_v[:,:,i])
    data_v_norm[:,:,i] = (data_v[:,:,i] - v_min)/(v_max - v_min)

    u_min_vec[i] = u_min
    u_max_vec[i] = u_max
    v_min_vec[i] = v_min
    v_max_vec[i] = v_max

  data_u = data_u_norm
  data_v = data_v_norm
  plt.plot(data_v_norm.reshape(-1))

  # print(u.shape)
  u_data = data_u[:,:,:].reshape((N_x+1)*(N_y+1),N_t)
  v_data = data_v[:,:,:].reshape((N_x+1)*(N_y+1),N_t)
  # b_data = np.tile(b[:,:,0],(1,N_t,1,1)).squeeze().reshape(N_t,(N_x+1)*(N_y+1)).T
  # print('original u',u.shape,u[:,:,0],'\nu.reshape((N_x+1)*(N_y+1),N_t)',u_data.shape,u_data)
  u0_data = data_u[:,:,0].reshape((N_x+1)*(N_y+1),1)
  v0_data = data_v[:,:,0].reshape((N_x+1)*(N_y+1),1)

  t_data = t_bounds[:N_t]
  # print(t_bounds)
  t_data = np.tile(t_data,((N_x+1)*(N_y+1),1))


  x_data = x_bounds.reshape(-1,1)
  x_data = np.tile(x_data, (1, N_x+1))
  x_data = np.reshape(x_data, (-1, 1))
  x_data = np.tile(x_data, (1, N_t))

  y_data = y_bounds.reshape((1,-1)) #Note this reshape is (1,-1) and NOT (-1,1)
  y_data = np.tile(y_data, ((N_y+1), 1))
  y_data = np.reshape(y_data, (-1, 1))
  y_data = np.tile(y_data, (1, N_t))

  # print(x_data,'\n',y_data)
  N_s = 1000
  steps = 40
  print('N_s and N_t are:',N_s,N_t)
  idx_s = np.random.choice(x_data.shape[0], N_s, replace = False)
  # idx_y = np.random.choice(x_bounds.shape[0], N_s, replace = False)
  idx_t = np.random.choice(N_t,steps, replace = False)
  print(idx_t,'\n\n',idx_s)
  # print(idx_t)
  u_max = np.tile(u_max_vec,(((N_x+1)*(N_y+1)),1)).squeeze().reshape((N_x+1)*(N_y+1),N_t)
  u_min = np.tile(u_min_vec,(((N_x+1)*(N_y+1)),1)).squeeze().reshape((N_x+1)*(N_y+1),N_t)
  v_max = np.tile(v_max_vec,(((N_x+1)*(N_y+1)),1,1,1)).squeeze().reshape((N_x+1)*(N_y+1),N_t)
  v_min = np.tile(v_min_vec,(((N_x+1)*(N_y+1)),1,1,1)).squeeze().reshape((N_x+1)*(N_y+1),N_t)


  t_meas = t_data[idx_s, :]
  t_meas = t_meas[:, idx_t].reshape((-1,1))
  x_meas = x_data[idx_s, :]
  x_meas = x_meas[:, idx_t].reshape((-1,1))
  y_meas = y_data[idx_s, :]
  y_meas = y_meas[:, idx_t].reshape((-1,1))
  u_meas = u_data[idx_s, :]
  u_meas = u_meas[:, idx_t].reshape((-1,1))
  v_meas = v_data[idx_s, :]
  v_meas = v_meas[:, idx_t].reshape((-1,1))

  u_max_meas = u_max[idx_s,:][:,idx_t].reshape((-1,1))
  u_min_meas = u_min[idx_s,:][:,idx_t].reshape((-1,1))
  v_max_meas = v_max[idx_s,:][:,idx_t].reshape((-1,1))
  v_min_meas = v_min[idx_s,:][:,idx_t].reshape((-1,1))

  # print(u_max_meas[:5,:6])

  X_meas = np.hstack((x_meas, y_meas, t_meas))
  # print(X_meas,'\n','\n',u_meas.T)
  # print(x_bounds,'\n',u[:,:,0])

  Split_TrainVal = 0.8
  N_train = int(N_s*steps*Split_TrainVal)
  idx_train = np.random.choice(X_meas.shape[0], N_train, replace=False)
  X_train = X_meas[idx_train,:]
  u_train = u_meas[idx_train,:]
  v_train = v_meas[idx_train,:]
  u_max_train = u_max_meas[idx_train,:]
  u_min_train = u_min_meas[idx_train,:]
  v_max_train = v_max_meas[idx_train,:]
  v_min_train = v_min_meas[idx_train,:]
  # Validation Measurements, which are the rest of measurements
  idx_val = np.setdiff1d(np.arange(X_meas.shape[0]), idx_train, assume_unique=True)
  X_val = X_meas[idx_val,:]
  u_val = u_meas[idx_val,:]
  v_val = v_meas[idx_val,:]
  u_max_val = u_max_meas[idx_val,:]
  u_min_val = u_min_meas[idx_val,:]
  v_max_val = v_max_meas[idx_val,:]
  v_min_val = v_min_meas[idx_val,:]

  N_s_col = 10
  steps_col = 5
  idx_s_col = np.random.choice(x_data.shape[0], N_s, replace = True)
  # idx_y = np.random.choice(x_bounds.shape[0], N_s, replace = False)
  idx_t_col = np.random.choice(N_t-10,steps, replace = True)
  # print(idx_t)

  t_meas_col = t_data[idx_s, :]
  t_meas_col = t_meas_col[:, idx_t].reshape((-1,1))
  x_meas_col = x_data[idx_s, :]
  x_meas_col = x_meas_col[:, idx_t].reshape((-1,1))
  y_meas_col = y_data[idx_s, :]
  y_meas_col = y_meas_col[:, idx_t].reshape((-1,1))
  u_meas_col = u_data[idx_s, :]
  u_meas_col = u_meas_col[:, idx_t].reshape((-1,1))
  v_meas_col = v_data[idx_s, :]
  v_meas_col = v_meas_col[:, idx_t].reshape((-1,1))

  X_meas_col = np.hstack((x_meas_col, y_meas_col, t_meas_col))


  X_train_col = X_meas_col[idx_train,:]
  u_train_col = u_meas_col[idx_train,:]
  v_train_col = v_meas_col[idx_train,:]

  def count_weights_and_nnz(net):
    total_weights = 0
    total_nnz = 0

    layers = [
        (net.hidden_layer1, net.g1),
        (net.hidden_layer2, net.g2),
        (net.hidden_layer3, net.g3),
        (net.hidden_layer4, net.g4),
        (net.hidden_layer5, net.g5),
        (net.hidden_layer6, net.g6),
    ]

    for layer, gate in layers:
        w = layer.weight
        b = layer.bias

        in_features = w.shape[1]
        out_features = w.shape[0]

        # Count total weights in this layer
        total_layer = w.numel() + b.numel()
        total_weights += total_layer

        # Expected active neurons from gate (soft L0)
        s = torch.sigmoid(gate.qz_loga).detach()
        active = (s > 0.5).float()  # hard threshold
        n_active = int(active.sum().item())

        # Each active neuron has all its incoming weights + 1 bias
        nnz_layer = n_active * (in_features + 1)
        total_nnz += nnz_layer

    # Output layer (fully dense)
    w = net.output_layer.weight
    b = net.output_layer.bias
    total_weights += w.numel() + b.numel()
    total_nnz += w.numel() + b.numel()

    return total_weights, total_nnz

  def pinn_dict(u,u_x,u_xx,b,b_x):
    ones = torch.ones(u.shape[0])
    dict_list = [u,u_x,
                u_xx,b,b_x]
    m = u.shape[0]
    # print('SHAPE IS',u_x.shape,v1.shape)
    n = 1
    X = torch.zeros((m,int(len(dict_list)*n)))
    for i in range(1,int(len(dict_list))):
      X[:,i] = dict_list[i].reshape(-1)
    # print(X.shape)
    return X.to(device)

  def m(x):
    return torch.sin(x)
  def compute_residuals(u, v, x, y, t, umin, umax, vmin, vmax, L):
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True,allow_unused=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x , create_graph=True,allow_unused=True)[0]
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True,allow_unused=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y , create_graph=True,allow_unused=True)[0]
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True,allow_unused=True)[0]
    v_t = torch.autograd.grad(v.sum(), t, create_graph=True,allow_unused=True)[0]
    # print(b.shape,u.shape)
    v_x = torch.autograd.grad(v.sum(), x, create_graph=True,allow_unused=True)[0]
    v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True,allow_unused=True)[0]
    v_y = torch.autograd.grad(v.sum(), y, create_graph=True,allow_unused=True)[0]
    v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=True,allow_unused=True)[0]
    c_v = vmax-vmin
    c_u = umax - umin

    # pde_u = u_t - kappa*(u_yy + u_xx) + v1.reshape(-1,1)*u_x + v2.reshape(-1,1)*u_y  - F_out.reshape(-1,1)
    pde_u = (u_xx + u_yy + u - u**3 - v + 0.01 - u_t).reshape(-1,1)
    # pde_v = (100*v_xx + 100*v_yy + 0.25*u - 0.25*v -v_t).reshape(-1,1)
    pde_v = (100*c_v*(1/L**2)*v_xx + 100*c_v*(1/L**2)*v_yy + 0.25*c_u*u + 0.25*umin - 0.25*c_v*v - 0.25*vmin -c_v*(1/50)*v_t).reshape(-1,1)
    return pde_u,pde_v
  
  # class L0Gate(nn.Module):
  #   def __init__(self, shape, droprate_init=0.5, temperature=2./3.):
  #       super().__init__()
  #       self.qz_loga = nn.Parameter(torch.Tensor(shape))
  #       self.temperature = temperature
  #       # init log-alpha
  #       self.qz_loga.data.normal_(mean=np.log(droprate_init) - np.log(1 - droprate_init), std=1e-2)

  #   def _hard_concrete_sample(self):
  #       u = torch.rand_like(self.qz_loga)
  #       s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.qz_loga) / self.temperature)
  #       z = s* (1.1 - 0.1) + 0.1  # Stretch to (0.1, 1.1)
  #       return torch.clamp(z, 0, 1)

  #   def forward(self):
  #       return self._hard_concrete_sample()

  #   def l0_loss(self):
  #       # Expected gate value → expected L0 norm
  #       s = torch.sigmoid(self.qz_loga)
  #       return torch.sum(s)
  
  class Net(nn.Module):
    def __init__(self, H):
        super(Net, self).__init__()
        
        self.hidden_layer1 = nn.Linear(3, H)
        self.hidden_layer2 = nn.Linear(H, H)
        self.hidden_layer3 = nn.Linear(H, H)
        self.hidden_layer4 = nn.Linear(H, H)
        self.hidden_layer5 = nn.Linear(H, H)
        self.hidden_layer6 = nn.Linear(H, H)

        # Add gates (one per neuron)
        self.g1 = L0Gate((H,))
        self.g2 = L0Gate((H,))
        self.g3 = L0Gate((H,))
        self.g4 = L0Gate((H,))
        self.g5 = L0Gate((H,))
        self.g6 = L0Gate((H,))

        self.output_layer = nn.Linear(H, 2)

    def forward(self, x, y, t, umin, umax, vmin, vmax):
        inputs = torch.cat([x,y,t],axis=1)

        z1 = self.g1()
        z2 = self.g2()
        z3 = self.g3()
        z4 = self.g4()
        z5 = self.g5()
        z6 = self.g6()

        h1 = m(self.hidden_layer1(inputs)) * z1
        h2 = m(self.hidden_layer2(h1)) * z2
        h3 = m(self.hidden_layer3(h2)) * z3
        h4 = m(self.hidden_layer4(h3)) * z4
        h5 = m(self.hidden_layer5(h4)) * z5
        h6 = m(self.hidden_layer6(h5)) * z6

        output = self.output_layer(h6)
        u = output[:,0].reshape(-1,1)
        v = output[:,1].reshape(-1,1)
        pdeu=0
        pdev=0

        return u, v, pdeu, pdev


  # class Net(nn.Module):
  #     def __init__(self,H):
  #         super(Net, self).__init__()
  #         self.hidden_layer1 = nn.Linear(3,H)
  #         self.hidden_layer2 = nn.Linear(H,H)
  #         self.hidden_layer3 = nn.Linear(H,H)
  #         self.hidden_layer4 = nn.Linear(H,H)
  #         self.hidden_layer5 = nn.Linear(H,H)
  #         self.hidden_layer6 = nn.Linear(H,H)
  #         self.output_layer = nn.Linear(H,2)



  #     def forward(self, x,y,t,umin,umax,vmin,vmax):
  #         inputs = torch.cat([x,y,t],axis=1) # combined two arrays of 1 columns each to one array of 3 columns
  #         # print(inputs)
  #         layer1_out = m(self.hidden_layer1(inputs))
  #         layer2_out = m(self.hidden_layer2(layer1_out))
  #         layer3_out = m(self.hidden_layer3(layer2_out))
  #         layer4_out = m(self.hidden_layer4(layer3_out))
  #         layer5_out = m(self.hidden_layer5(layer4_out))
  #         layer6_out = m(self.hidden_layer6(layer5_out))
  #         output = self.output_layer(layer6_out)
  #         u = output[:,0].reshape(-1,1)
  #         v = output[:,1].reshape(-1,1)
  #         # print(u.requires_grad)
  #         # u_x = torch.autograd.grad(u.sum(), x, create_graph=True,allow_unused=True)[0]
  #         # u_xx = torch.autograd.grad(u_x.sum(), x , create_graph=True,allow_unused=True)[0]
  #         # u_y = torch.autograd.grad(u.sum(), y, create_graph=True,allow_unused=True)[0]
  #         # u_yy = torch.autograd.grad(u_y.sum(), y , create_graph=True,allow_unused=True)[0]
  #         # u_t = torch.autograd.grad(u.sum(), t, create_graph=True,allow_unused=True)[0]
  #         # v_t = torch.autograd.grad(v.sum(), t, create_graph=True,allow_unused=True)[0]
  #         # # print(b.shape,u.shape)
  #         # v_x = torch.autograd.grad(v.sum(), x, create_graph=True,allow_unused=True)[0]
  #         # v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True,allow_unused=True)[0]
  #         # v_y = torch.autograd.grad(v.sum(), y, create_graph=True,allow_unused=True)[0]
  #         # v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=True,allow_unused=True)[0]
  #         # c_v = vmax-vmin
  #         # c_u = umax - umin

  #         # pde_u = u_t - kappa*(u_yy + u_xx) + v1.reshape(-1,1)*u_x + v2.reshape(-1,1)*u_y  - F_out.reshape(-1,1)
  #         pde_u = 0 #(u_xx + u_yy + u - u**3 - v + 0.01 - u_t).reshape(-1,1)
  #         # pde_v = (100*v_xx + 100*v_yy + 0.25*u - 0.25*v -v_t).reshape(-1,1)
  #         pde_v = 0 #(100*c_v*(1/L**2)*v_xx + 100*c_v*(1/L**2)*v_yy + 0.25*c_u*u + 0.25*umin - 0.25*c_v*v - 0.25*vmin -c_v*(1/50)*v_t).reshape(-1,1)

  #         # pde_u = (0.1*u_xx + 0.1*u_yy - u*v**2 - u**3 + v**3 +u**2*v + u - u_t).reshape(-1,1)
  #         # pde_v = (0.1*v_xx + 0.1*v_yy - u*v**2 - u**3 - v**3 -u**2*v - u -v_t).reshape(-1,1)
  #         return u,v,pde_u,pde_v


  mse_cost_function1 = torch.nn.MSELoss() # Mean squared error

  # plt.plot(X_train[:,2],'o')

  def cart_inputs(x,y,t):
    a = np.array([[x0, y0,t0] for x0 in x for y0 in y for t0 in t])
    return a[:,0].reshape(-1,1), a[:,1].reshape(-1,1), a[:,2].reshape(-1,1)

  # x_ic,y_ic,t_ic = cart_inputs(x_bounds,y_bounds,np.zeros((1)))
  x_ic,y_ic,t_ic = cart_inputs(x_bounds,y_bounds,t_bounds[0]*np.ones((1)))

  u_ic = torch.tensor(u0_data).reshape(-1).reshape(-1,1).detach().numpy()
  v_ic = torch.tensor(v0_data).reshape(-1).reshape(-1,1).detach().numpy()

  x_collocation = X_train[:,0].reshape(-1,1) #np.random.uniform(low=x_min, high=x_max, size=(N_x+1,1))
  y_collocation = X_train[:,1].reshape(-1,1)
  t_collocation = X_train[:,2].reshape(-1,1)
  pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True).to(device)
  pt_y_collocation = Variable(torch.from_numpy(y_collocation).float(), requires_grad=True).to(device)
  pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True).to(device)

  x_collocation1 = X_train_col[:,0].reshape(-1,1) #np.random.uniform(low=x_min, high=x_max, size=(N_x+1,1))
  y_collocation1 = X_train_col[:,1].reshape(-1,1)
  t_collocation1 = X_train_col[:,2].reshape(-1,1)
  pt_x_collocation1 = Variable(torch.from_numpy(x_collocation1).float(), requires_grad=True).to(device)
  pt_y_collocation1 = Variable(torch.from_numpy(y_collocation1).float(), requires_grad=True).to(device)
  pt_t_collocation1 = Variable(torch.from_numpy(t_collocation1).float(), requires_grad=True).to(device)

  pt_x_ic = Variable(torch.from_numpy(x_ic).float(), requires_grad=True).to(device)
  pt_y_ic = Variable(torch.from_numpy(y_ic).float(), requires_grad=True).to(device)
  pt_t_ic = Variable(torch.from_numpy(t_ic).float(), requires_grad=True).to(device)

  all_zeros_col = np.zeros((X_train_col.shape[0],1))
  pt_all_zeros_col = Variable(torch.from_numpy(all_zeros_col).float(), requires_grad=False).to(device)

  all_zeros = np.zeros((X_train.shape[0],1))
  pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)

  x_val = X_val[:,0].reshape(-1,1) #np.random.uniform(low=x_min, high=x_max, size=(N_x+1,1))
  y_val = X_val[:,1].reshape(-1,1) #np.random.uniform(low=y_min, high=y_max, size=(N_y+1,1))
  t_val = X_val[:,2].reshape(-1,1)

  pt_x_val = Variable(torch.from_numpy(x_val).float(), requires_grad=True).to(device)
  pt_y_val = Variable(torch.from_numpy(y_val).float(), requires_grad=True).to(device)
  pt_t_val = Variable(torch.from_numpy(t_val).float(), requires_grad=True).to(device)


  # === Hyperparameters ===
  learning_rates = [5e-03]
  # hidden_dims = [200]
  hidden_dim = 100
  lam_0 = [1e-2,1e-4]
  num_repeats = 3
  validate_every = 5000
  max_epochs = 100000
  patience = 5

  best_global_val_loss = float('inf')
  best_model_state = None
  best_hparams = {}

  results = []

  for lr in learning_rates:
      # for hidden_dim in hidden_dims:
      for lambda0 in lam_0:
          run_errors = []

          print(f"\n=== Training with lr={lr}, L0={lambda0} ===")

          for run in range(num_repeats):
              print(f"Run {run + 1}/{num_repeats}")
              torch.manual_seed(run)
              class L0Gate(nn.Module):
                def __init__(self, shape, droprate_init=0.5, temperature=2./3.):
                    super().__init__()
                    self.qz_loga = nn.Parameter(torch.Tensor(shape))
                    self.temperature = temperature
                    # init log-alpha
                    self.qz_loga.data.normal_(mean=np.log(droprate_init) - np.log(1 - droprate_init), std=1e-2)

                def _hard_concrete_sample(self):
                    u = torch.rand_like(self.qz_loga)
                    s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.qz_loga) / self.temperature)
                    z = s* (1.1 - 0.1) + 0.1  # Stretch to (0.1, 1.1)
                    return torch.clamp(z, 0, 1)

                def forward(self):
                    return self._hard_concrete_sample()

                def l0_loss(self):
                    # Expected gate value → expected L0 norm
                    s = torch.sigmoid(self.qz_loga)
                    return torch.sum(s)

              t_start_train = time.time()
              net = Net(hidden_dim).to(device)
              optimizer = torch.optim.Adam(net.parameters(), lr=lr)

              best_val_loss = float('inf')
              best_model_wts = copy.deepcopy(net.state_dict())
              epochs_no_improve = 0

              for epoch in range(0, max_epochs):
                  net.train()
                  optimizer.zero_grad()

                  uout, vout, f_out1, g_out1 = net(
                      pt_x_collocation, pt_y_collocation, pt_t_collocation,
                      torch.tensor(u_min_train).float().to(device),
                      torch.tensor(u_max_train).float().to(device),
                      torch.tensor(v_min_train).float().to(device),
                      torch.tensor(v_max_train).float().to(device)
                  )
                  f_out, g_out = compute_residuals(
                      uout, vout,
                      pt_x_collocation, pt_y_collocation, pt_t_collocation,
                      torch.tensor(u_min_train).float().to(device),
                      torch.tensor(u_max_train).float().to(device),
                      torch.tensor(v_min_train).float().to(device),
                      torch.tensor(v_max_train).float().to(device),
                      L
                  )

                  net_uic, net_vic, _, _ = net(
                      pt_x_ic, pt_y_ic, pt_t_ic,
                      torch.tensor(u_min[:, 0]).float().to(device),
                      torch.tensor(u_max[:, 0]).float().to(device),
                      torch.tensor(v_min[:, 0]).float().to(device),
                      torch.tensor(v_max[:, 0]).float().to(device)
                  )

                  u_out = uout.reshape(-1, 1)
                  v_out = vout.reshape(-1, 1)

                  mse_v = mse_cost_function1(g_out, pt_all_zeros)
                  mse_vic = mse_cost_function1(net_vic.reshape(-1, 1), torch.from_numpy(v_ic.reshape(-1, 1)).float().to(device))
                  mse_udata = mse_cost_function1(u_out, torch.tensor(u_train).float().to(device))
                  L0_term = (net.g1.l0_loss() + net.g2.l0_loss() + net.g3.l0_loss() + net.g4.l0_loss() 
                             + net.g5.l0_loss() + net.g6.l0_loss())

                  # Weight the regularization
                  # lambda_l0 = 1e-5  # tune this


                  loss = mse_udata + mse_v + mse_vic + lambda0 * L0_term
                  # loss = mse_udata # + mse_vic
                  loss.backward()
                  optimizer.step()

                  if epoch % validate_every == 0:
                      net.eval()
                      # with torch.no_grad():
                      u_outval, v_outval, _, _ = net(
                          pt_x_val, pt_y_val, pt_t_val,
                          torch.tensor(u_min_val).float().to(device),
                          torch.tensor(u_max_val).float().to(device),
                          torch.tensor(v_min_val).float().to(device),
                          torch.tensor(v_max_val).float().to(device)
                      )

                      rel_u = torch.norm(torch.tensor(u_val).reshape(-1).to(device) - u_outval.reshape(-1)) / \
                              torch.norm(torch.tensor(u_val).reshape(-1).to(device))

                      rel_v = torch.norm(torch.tensor(v_val).reshape(-1).to(device) - v_outval.reshape(-1)) / \
                              torch.norm(torch.tensor(v_val).reshape(-1).to(device))

                      val_loss = 0.5 * (rel_u + rel_v)
                      # val_loss = rel_u
                      print(val_loss)

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
              # Load best weights
              net.load_state_dict(best_model_wts)

              # Evaluate final validation score
              s=1
              xx= x_bounds[::s] #np.linspace(x_min,x_max,41)
              yy= y_bounds[::s] #np.linspace(x_min,x_max,41)
              tt= t_data[0,:] #np.linspace(0,10,100)
              x1,y1,t1 = cart_inputs(xx,yy,tt)
              pt_x = Variable(torch.from_numpy(x1).float(), requires_grad=True).to(device)
              pt_y = Variable(torch.from_numpy(y1).float(), requires_grad=True).to(device)
              pt_t = Variable(torch.from_numpy(t1).float(), requires_grad=True).to(device)
              t_start_eval = time.time()
              pt_u,pt_v,_,_ = net(pt_x,pt_y,pt_t,torch.tensor(u_min[::s,::s]).reshape(-1,1).float().to(device),
                                              torch.tensor(u_max[::s,::s]).reshape(-1,1).float().to(device),torch.tensor(v_min[::s,::s]).reshape(-1,1).float().to(device),
                                              torch.tensor(v_max[::s,::s]).reshape(-1,1).float().to(device)) #.detach().numpy()
              t_end_eval = time.time()
              ms_u = pt_u.reshape(xx.shape[0],yy.shape[0],tt.shape[0])
              ms_v = pt_v.reshape(xx.shape[0],yy.shape[0],tt.shape[0])
              full_field_true = torch.hstack((torch.tensor(data_u_norm_clean),torch.tensor(data_v_norm_clean))).to(device)
              full_field_net = torch.hstack((ms_u,ms_v)).detach()

              error_uv = torch.zeros(N_t)
              error_v = torch.zeros(N_t)
              error_u = torch.zeros(N_t)
              for i in range(N_t):
                error_uv[i] = torch.norm(full_field_true[:,:,i] - full_field_net[:,:,i])/torch.norm(full_field_true[:,:,i])
                error_v[i] = torch.norm(torch.tensor(data_v_norm_clean[::s,::s,i]).to(device) - ms_v[:,:,i])/torch.norm(torch.tensor(data_v_norm_clean[::s,::s,i]).to(device))
                error_u[i] = torch.norm(torch.tensor(data_u_norm_clean[::s,::s,i]).to(device) - ms_u[:,:,i])/torch.norm(torch.tensor(data_u_norm_clean[::s,::s,i]).to(device))

              print('\nError uv',torch.mean(error_uv),' Error u',torch.mean(error_u),'Error v:', torch.mean(error_v),'\n')

              final_error = torch.mean(error_uv)
              run_errors.append(final_error.item())

              print(f"Run {run+1}/{num_repeats} Final Val Error: {final_error.item():.6f}")
              total_w, nnz_w = count_weights_and_nnz(net)
              perc = 100 * nnz_w / total_w

              print(f"\nTotal weights: {total_w}")
              print(f"\nNon-zero (active) weights: {nnz_w}")
              print(f"\nPercentage active: {perc:.2f}%")
              print(f"\nTotal training time is",t_end_train-t_start_train)
              print(f"\nTotal Evaluation time is", t_end_eval - t_start_eval)
              print("\n---------------------------------------------------------\n")

          # Compute mean ± 95% CI
          mean_error = np.mean(run_errors)
          sem = stats.sem(run_errors)
          ci95 = sem * stats.t.ppf((1 + 0.95) / 2., num_repeats - 1)

          print(f"At lr={lr}, hidden_dim={hidden_dim} mean full-field L2 Error: {mean_error:.6f} ± {ci95:.6f}")
          # torch.save(net,f'/home/esaha/links/scratch/L0-trained-models-outputs/fhn-L0-{N_s}Ns-lr-{lr}-L0-{lambda0}-4Layers-{hidden_dim}dim')
          results.append((lr, hidden_dim, mean_error, ci95))

          if mean_error < best_global_val_loss:
              best_global_val_loss = mean_error
              best_model_state = copy.deepcopy(net.state_dict())
              best_hparams = {'lr': lr, 'hidden_dim': hidden_dim}

  # Final results
  # for r in results:
  #     print(f"lr={r[0]:.0e}, hidden_dim={r[1]:>3} → Val Error: {r[2]:.6f} ± {r[3]:.6f}")

  print(f"Best Hyperparameters: {best_hparams}, Validation Error: {best_global_val_loss:.6f}")
  torch.save(net,f'/home/esaha/links/scratch/L0-trained-models-outputs/fhn-{N_s}Ns-lr-{lr}-L0-{lambda0}-4Layers-{hidden_dim}dim-Best')