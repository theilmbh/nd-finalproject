import numpy as np
import scipy as sp
from scipy import signal
from scipy.io import wavfile
from numpy import linalg as LA
import mpl_toolkits.mplot3d
from scipy.integrate import odeint
import pylab as plt


# Authors: Tammy Tran, Nuttida Rungratsameetaweemana, Brad Theilman

tau = 10  # time constant
p = 0.1 # internal connectivity

Ng = 1000.0 # Network Size

lambd = 1.2 # chaoticity level

train_dur = 1000*tau

# noise in the firing rate
theta_state = 0.05
def zeta_state(t):
    return np.random.uniform(low = -theta_state, high = theta_state)

# exploration noise
def zeta(t):
    return np.random.uniform(low = -0.5, high = 0.5)

# decaying learning rate
eta_init = 0.0005
T = 20 #s
def eta(t):
    return eta_init/(1 + (t/T))

tau_avg = 5 #ms

# Heaviside
def heaviside(s, intermediate):
    if s < 0:
        return 0.0
    if s > 0:
        return 1.0
    if s == 0:
        return intermediate

# Exponential Filter
tau_l = 50 #ms
def g(s):
    return sp.exp(-s/tau_l)*heaviside(s, 1)

# H smoothing functio
def h(s):
    return heaviside(s, 1) - heaviside(s - 100, 1)


# input streams
def u_on(t):
    return np.random.random_sample((sp.size(t),)) < 0.0005
def u_off(t):
    return np.random.random_sample((sp.size(t),)) < 0.0005
#def u_on_steam(t):
#def u_off_steam(t):
    
# target function
f_prev = 0
def f(uon, uoff):
    f_prev = 0
    uon_len = sp.size(uon)
    uoff_len = sp.size(uoff)
    if uon_len != uoff_len:
        return 0

    fret = sp.zeros(uon_len)
    for samp in sp.arange(1, uon_len):
        if uon[samp] == 1:
            f_prev = 0.5
        if uoff[samp] == 1:
            f_prev = -0.5
        fret[samp] = f_prev
    return fret

def f_actual(uon, uoff):
    # 1/ sigma f times f(t) * g, where * is a convolution operation, discretized by t_step
    return f(uon, uoff)    

# firing rate
def r(t, x):
    return sp.tanh(x) + zeta_state(t)

sigma_p = np.sqrt(1/(p * Ng)) #why is there a square root?
sigma_w = np.sqrt(1/(Ng))
 
W_rec = sigma_p * np.random.randn(Ng, Ng) #weights in recurrent network
W_in = np.random.uniform(-1, 1, Ng)     #weights for input
W_fb = np.random.uniform(-1, 1, Ng) #weights for feedback
W_rec[np.random.random_sample((Ng, Ng)) > p] = 0

w = np.random.randn(Ng)*sigma_w #weights for output
x = np.random.randn(Ng)*sigma_w #random initial x for network

P_avg = 0
z_avg = 0

print("Creating statevector")
svinit =  np.concatenate((x, w))
print("Done")


def dNeurons(statevec, t, param):
    
    # Extract relevant parameters from the statevector

    print(t)
    dt = param[0]
    training = param[1]
    f_target = param[2]
    
    x_i = statevec[0:Ng]
    w_i = statevec[Ng:2*Ng]
    
    # Compute Firing Rates and feedback signals
    r_i = r(t, x_i)
    z_i = np.dot(w_i, r_i) + zeta(t)
    
    # Get uon and uoff for this time step
    uon = u_on(t)
    uoff = u_off(t)

    # Compute input signals based on raw uon/uoff values (low pass filter)

    # compute target output 
    target = f_target(uon, uoff)

    # compute if we are in training epoch or not
    train = training > 0 and t > training and t < training + train_dur

    # Compute next timestep depending on if training or not
    if train:
        
        u = u(t)
        # target = f(t)
        dxidt = (-x_i + lambd * np.dot(W_rec, r_i) + np.dot(W_in, u)
                 + np.dot(W_fb, z_i))/tau
        x_new = x_i + dxidt*dt
        r_new = r(t, x_new)
        
        P = -(z_i - target)**2
        M = P > P_avg
        
        dwdt = eta(t) * (z_i - z_avg) * M * r_i
        w_new = w_i + dwdt*dt
        
        z_new = np.dot(w_new, r_new)

        P_avg = (1 - (dt/tau_avg)) * P_avg + (dt/tau_avg) * P
        z_avg = (1 - (dt/tau_avg)) * z_i + (dt/tau_avg) * z_new
        f_prev = target
        
    else:
        dxidt = (-x_i + lambd * np.dot(W_rec, r_i) + np.dot(W_in, u(t))
                 + np.dot(W_fb, z_i))/tau
        x_new = x_i + dxidt*dt
        dwdt = np.zeros(np.shape(w_i))
        w_new = w_i
    
    return np.concatenate((x_new, w_new))
    
# Run the network
dt = 1
pre_train_dur = 0
post_train_dur = 8000
dur = pre_train_dur + train_dur + post_train_dur
times = sp.arange(0.0, dur, dt)
samps = np.size(times)

training = pre_train_dur

# generate input patterns
uon_t = u_on(times)
uoff_t = u_off(times)
f_raw = f(uon_t, uoff_t)



params = (dt, training, f_actual)
xsave = np.zeros((samps, Ng))
wsave = np.zeros((samps, Ng))


for indx, t in enumerate(times):
    xsave[indx, :] = svinit[0:Ng]
    wsave[indx, :] = svinit[Ng:2*Ng]
    svinit = dNeurons(svinit, t, params)

rates = np.tanh(xsave)

rate_pre = rates - np.mean(rates, 0)
rate_pre /= np.std(rate_pre, 0)
rate_cov = np.cov(rates, rowvar=0)
w, v = LA.eig(rate_cov)

prj1 = np.dot(rate_pre, v[:,0])
prj2 = np.dot(rate_pre, v[:,1])
prj3 = np.dot(rate_pre, v[:,2])

plt.figure().gca(projection='3d')
plt.plot(prj1, prj2, prj3)

z = np.sum(wsave*rates, axis=1)
plt.figure()
plt.plot(times, z)

plt.show()
