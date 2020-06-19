## http://weinbe58.github.io/QuSpin/generated/quspin.basis.boson_basis_general.html#quspin.basis.boson_basis_general
## http://weinbe58.github.io/QuSpin/generated/quspin.operators.hamiltonian.html

from __future__ import print_function, division
from quspin.operators import hamiltonian # operators
from quspin.basis import boson_basis_general # spin basis constructor
import numpy as np # general math functions
#
###### define model parameters ######
Lx, Ly = 3, 3 # linear dimension of spin 1 2d lattice
N_2d = Lx*Ly # number of sites for spin 1
N_sps = 3
#
#J=0.1 # hopping matrix element
U=1.0 # onsite interaction
mu=0.371 # chemical potential
#
###### setting up user-defined symmetry transformations for 2d lattice ######
s = np.arange(N_2d) # sites [0,1,2,....]
x = s%Lx # x positions for sites
y = s//Lx # y positions for sites
T_x = (x+1)%Lx + Lx*y # translation along x-direction
T_y = x +Lx*((y+1)%Ly) # translation along y-direction
P_x = x + Lx*(Ly-y-1) # reflection about x-axis
P_y = (Lx-x-1) + Lx*y # reflection about y-axis
#
###### setting up bases ######
basis_2d = boson_basis_general(N_2d,sps=N_sps,kxblock=(T_x,0),kyblock=(T_y,0),pxblock=(P_x,0),pyblock=(P_y,0))
#
###### setting up hamiltonian ######
# setting up site-coupling lists
#hopping=[[-J,i,T_x[i]] for i in range(N_2d)] + [[-J,i,T_y[i]] for i in range(N_2d)]
potential=[[-mu-U/2.0,i] for i in range(N_2d)]
interaction=[[U/2.0,i,i] for i in range(N_2d)]
#
print("# J E/N")
for J in np.linspace(0,0.1,41): # hopping matrix element
    hopping=[[-J,i,T_x[i]] for i in range(N_2d)] + [[-J,i,T_y[i]] for i in range(N_2d)]
    static=[["+-",hopping],["-+",hopping],["n",potential],["nn",interaction]]
# build hamiltonian
    no_checks=dict(check_symm=False, check_pcon=False, check_herm=False)
#    H=hamiltonian(static,[],basis=basis_2d,dtype=np.float64)
#    H=(hamiltonian(static,[],basis=basis_2d,dtype=np.float64)).as_sparse_format(static_fmt="csr")
#    H=hamiltonian(static,[],static_fmt="dia",basis=basis_2d,dtype=np.float64)
#    H=hamiltonian(static,[],static_fmt="csr",basis=basis_2d,dtype=np.float64)
    H=hamiltonian(static,[],static_fmt="csr",basis=basis_2d,dtype=np.float64,**no_checks)
#    print(H)
# diagonalise H
#    E=H.eigvalsh()
#    print("# J E/N",J,E[0]/N_2d)
    ene,vec = H.eigsh(time=0.0,which='SA',k=2)
#    print("# J E/N",J,ene[0]/N_2d)
    print(J,ene[0]/N_2d)
