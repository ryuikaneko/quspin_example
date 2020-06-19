## https://weinbe58.github.io/QuSpin/examples/user-basis_example2.html#user-basis-example2-label
## https://weinbe58.github.io/QuSpin/downloads/567d8096559c83a92c52a580c93935c1/user_basis_trivial-boson.py

from __future__ import print_function, division
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import boson_basis_1d # Hilbert space spin basis_1d
import numpy as np
#
###### define model parameters ######
N=10 # lattice sites
N_sps=3 # states per site
#Nb=N//2 # total number of bosons
#Nb=N # total number of bosons
###### setting up bases ######
#basis_1d=boson_basis_1d(N,Nb=Nb,sps=N_sps,kblock=0,pblock=1) 
basis_1d=boson_basis_1d(N,sps=N_sps,kblock=0,pblock=1) 
#print(basis_1d)
###### setting up hamiltonian ######
#
#J=0.1 # hopping matrix element
U=1.0 # onsite interaction
mu=0.4 # chemical potential
#
#hopping=[[-J,j,(j+1)%N] for j in range(N)]
interaction=[[0.5*U,j,j] for j in range(N)]
#potential=[[-0.5*U,j] for j in range(N)]
potential=[[-mu-0.5*U,j] for j in range(N)]
#
print("# J E/N")
for J in np.linspace(0,0.3,31): # hopping matrix element
    hopping=[[-J,j,(j+1)%N] for j in range(N)]
    static=[["+-",hopping],["-+",hopping],["nn",interaction],["n",potential]]
    dynamic=[]
#
    no_checks=dict(check_symm=False, check_pcon=False, check_herm=False)
    H=hamiltonian(static,dynamic,static_fmt="csr",basis=basis_1d,dtype=np.float64,**no_checks)
#    print(H)
# diagonalise H
    ene,vec = H.eigsh(time=0.0,which='SA',k=2)
    print(J,ene[0]/N)
