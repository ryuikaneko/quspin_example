## https://weinbe58.github.io/QuSpin/examples/example7.html
## https://weinbe58.github.io/QuSpin/examples/example15.html
## https://weinbe58.github.io/QuSpin/examples/user-basis_example0.html
## https://weinbe58.github.io/QuSpin/user_basis.html
## https://weinbe58.github.io/QuSpin/generated/quspin.basis.spin_basis_1d.html
from __future__ import print_function, division
from quspin.operators import hamiltonian # operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
import numpy as np # general math functions
#
###### define model parameters ######
Jleg = 1.0 # spin-spin interaction, leg
Jrung = 1.0 # spin-spin interaction, rung
L = 12 # length of chain
N = 2*L # number of sites
###### setting up bases ######
#basis_1d = spin_basis_1d(L=N,Nup=N//2,S="1/2",pauli=0)
basis_1d = spin_basis_1d(L=N,Nup=N//2,S="1/2",pauli=0,a=2,kblock=0,pblock=1,zblock=1)## even L
#basis_1d = spin_basis_1d(L=N,Nup=N//2,S="1/2",pauli=0,a=2,kblock=0,pblock=-1,zblock=-1)## odd L
###### setting up hamiltonian ######
Jzzs = \
    [[Jleg,i,(i+2)%N] for i in range(0,N,2)] \
  + [[Jleg,i,(i+2)%N] for i in range(1,N,2)] \
  + [[Jrung,i,i+1] for i in range(0,N,2)]
Jpms = \
    [[0.5*Jleg,i,(i+2)%N] for i in range(0,N,2)] \
  + [[0.5*Jleg,i,(i+2)%N] for i in range(1,N,2)] \
  + [[0.5*Jrung,i,i+1] for i in range(0,N,2)]
Jmps = \
    [[0.5*Jleg,i,(i+2)%N] for i in range(0,N,2)] \
  + [[0.5*Jleg,i,(i+2)%N] for i in range(1,N,2)] \
  + [[0.5*Jrung,i,i+1] for i in range(0,N,2)]
static = [["zz",Jzzs],["+-",Jpms],["-+",Jmps]]
# build hamiltonian
#H = hamiltonian(static,[],static_fmt="csr",basis=basis_1d,dtype=np.float64)
no_checks = dict(check_symm=False, check_pcon=False, check_herm=False)
H = hamiltonian(static,[],static_fmt="csr",basis=basis_1d,dtype=np.float64,**no_checks)
# diagonalise H
#ene,vec = H.eigsh(time=0.0,which="SA",k=2)
ene = H.eigsh(which="SA",k=2,return_eigenvectors=False); ene = np.sort(ene)
print(Jleg,Jrung,N,ene[0]/N)
## 2-leg ladder (L=inf): -0.578043140180 (PhysRevB.89.094424, see also PhysRevB.54.R3714, PhysRevB.47.3196)
