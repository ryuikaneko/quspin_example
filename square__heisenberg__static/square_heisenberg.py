## http://weinbe58.github.io/QuSpin/generated/quspin.basis.spin_basis_general.html#quspin.basis.spin_basis_general
from __future__ import print_function, division
from quspin.operators import hamiltonian # operators
from quspin.basis import spin_basis_general # spin basis constructor
import numpy as np # general math functions
#
###### define model parameters ######
J = 1.0 # spin-spin interaction
Lx, Ly = 4, 4 # linear dimension of 2d lattice
N_2d = Lx*Ly # number of sites
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
basis_2d = spin_basis_general(N=N_2d,Nup=N_2d//2,S="1/2",pauli=0,kxblock=(T_x,0),kyblock=(T_y,0),pxblock=(P_x,0),pyblock=(P_y,0),zblock=(Z,0))
#basis_2d = spin_basis_general(N=N_2d,Nup=N_2d//2,S="1/2",pauli=0)
#
###### setting up hamiltonian ######
# setting up site-coupling lists
Jzzs = [[J,i,T_x[i]] for i in range(N_2d)]+[[J,i,T_y[i]] for i in range(N_2d)]
Jpms = [[0.5*J,i,T_x[i]] for i in range(N_2d)]+[[0.5*J,i,T_y[i]] for i in range(N_2d)]
Jmps = [[0.5*J,i,T_x[i]] for i in range(N_2d)]+[[0.5*J,i,T_y[i]] for i in range(N_2d)]
#
static = [["zz",Jzzs],["+-",Jpms],["-+",Jmps]]
# build hamiltonian
#H = hamiltonian(static,[],static_fmt="csr",basis=basis_2d,dtype=np.float64)
no_checks = dict(check_symm=False, check_pcon=False, check_herm=False)
H = hamiltonian(static,[],static_fmt="csr",basis=basis_2d,dtype=np.float64,**no_checks)
# diagonalise H
#ene,vec = H.eigsh(time=0.0,which="SA",k=2)
ene = H.eigsh(which="SA",k=2,return_eigenvectors=False); ene = np.sort(ene)
print(J,ene[0]/N_2d,ene[1]/N_2d)
