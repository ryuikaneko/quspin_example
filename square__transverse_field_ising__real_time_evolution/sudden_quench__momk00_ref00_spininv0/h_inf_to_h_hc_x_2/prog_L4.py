from __future__ import print_function, division
import sys,os

## http://weinbe58.github.io/QuSpin/generated/quspin.basis.spin_basis_general.html#quspin.basis.spin_basis_general

from quspin.operators import hamiltonian # operators
from quspin.basis import spin_basis_general # spin basis constructor
import numpy as np # general math functions
#
import scipy.sparse
import scipy.sparse.linalg
import scipy as scipy
#
#import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#
###### define model parameters ######
J=-1.0 # spin=spin interaction
#g=-0.1 # magnetic field strength
#g=-3.04438 # magnetic field strength
g=-3.04438*2.0 # magnetic field strength
J0=0.0 # spin=spin interaction
g0=-1.0 # magnetic field strength
#Lx, Ly = 3, 3 # linear dimension of spin 1 2d lattice
Lx, Ly = 4, 4 # linear dimension of spin 1 2d lattice
#Lx, Ly = 5, 5 # linear dimension of spin 1 2d lattice
N_2d = Lx*Ly # number of sites for spin 1
#
###### setting up user-defined symmetry transformations for 2d lattice ######
s = np.arange(N_2d) # sites [0,1,2,....]
x = s%Lx # x positions for sites
y = s//Lx # y positions for sites
T_x = (x+1)%Lx + Lx*y # translation along x-direction
T_y = x +Lx*((y+1)%Ly) # translation along y-direction
P_x = x + Lx*(Ly-y-1) # reflection about x-axis
P_y = (Lx-x-1) + Lx*y # reflection about y-axis
Z   = -(s+1) # spin inversion
#
###### setting up bases ######
basis_2d = spin_basis_general(N_2d,kxblock=(T_x,0),kyblock=(T_y,0),pxblock=(P_x,0),pyblock=(P_y,0),zblock=(Z,0))
#basis0_2d = spin_basis_general(N_2d,Nup=N_2d)
#
###### setting up hamiltonian ######
# setting up site-coupling lists
Jzz=[[J,i,T_x[i]] for i in range(N_2d)]+[[J,i,T_y[i]] for i in range(N_2d)]
gx =[[g,i] for i in range(N_2d)]
#gz =[[g0,i] for i in range(N_2d)]
Jzz0=[[J0,i,T_x[i]] for i in range(N_2d)]+[[J0,i,T_y[i]] for i in range(N_2d)]
gx0 =[[g0,i] for i in range(N_2d)]
#
static=[["zz",Jzz],["x",gx]]
#static0=[["z",gz]]
static0=[["zz",Jzz0],["x",gx0]]
# build hamiltonian
H=hamiltonian(static,[],basis=basis_2d,dtype=np.float64)
H0=hamiltonian(static0,[],basis=basis_2d,dtype=np.float64)
# cast to sparse hamiltonian
## http://weinbe58.github.io/QuSpin/generated/quspin.operators.hamiltonian.html
## H_dia=H.as_sparse_format(static_fmt="csr",dynamic_fmt={(func,func_args):"dia"})
## H_csr=H.tocsr(time=time)
H_csr = H.tocsr(time=0)
H0_csr = H0.tocsr(time=0)
print(H_csr)
print(H0_csr)
# diagonalise H
#E=H.eigvalsh()
ene,vec = scipy.sparse.linalg.eigsh(H_csr,which='SA',k=2)
print(ene)
print(vec)
ene0,vec0 = scipy.sparse.linalg.eigsh(H0_csr,which='SA',k=2)
print(ene0)
print(vec0)

timei = 0.0
#timef = 1.0
#timef = 6.0*np.pi/4.0
timef = 2.0
dt = 0.01
Nsteps = int(timef/dt+0.1)+1
#Nsteps = 101
ret = scipy.sparse.linalg.expm_multiply((-1j)*H_csr,vec0[:,0],start=timei,stop=timef,num=Nsteps,endpoint=True)
print(ret)
norm = [np.linalg.norm(ret[i])**2 for i in range(Nsteps)]
print(norm)
ene = [(np.dot(np.conjugate(ret[i]),H_csr.dot(ret[i])) / norm[i]).real for i in range(Nsteps)]
print(ene)
mx = [(np.dot(np.conjugate(ret[i]),H0_csr.dot(ret[i])) / norm[i]).real / (-N_2d) for i in range(Nsteps)]
print(mx)
time_steps = [timei+i*(timef-timei)/(Nsteps-1) for i in range(Nsteps)]
#time_steps_pi = [timei+i*(timef-timei)/(Nsteps-1)/np.pi*4.0 for i in range(Nsteps)]
print(time_steps)
#print(time_steps_pi)

fig10 = plt.figure()
fig10.suptitle("mx")
plt.plot(time_steps,mx)
#plt.plot(time_steps_pi,mx)
plt.xlabel("$t$")
#plt.xlabel("$t/(\pi/4)$")
fig10.savefig("fig_mx.png")
#plt.show()
