from __future__ import print_function, division
from quspin.operators import hamiltonian # operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
import numpy as np # general math functions

def exact_diag(J1,J2,N,twoSz):
###### setting up bases ######
## https://github.com/weinbe58/QuSpin/issues/324
    basis_1d = spin_basis_1d(L=N,Nup=N//2+twoSz,S="1/2",pauli=0)
###### setting up hamiltonian ######
    Jzzs = \
        [[J1,i,(i+1)%N] for i in range(N)] \
      + [[J2,i,(i+2)%N] for i in range(N)]
    Jpms = \
        [[0.5*J1,i,(i+1)%N] for i in range(N)] \
      + [[0.5*J2,i,(i+2)%N] for i in range(N)]
    Jmps = \
        [[0.5*J1,i,(i+1)%N] for i in range(N)] \
      + [[0.5*J2,i,(i+2)%N] for i in range(N)]
    static = [\
        ["zz",Jzzs],["+-",Jpms],["-+",Jmps],\
        ]
# build hamiltonian
#    H = hamiltonian(static,[],static_fmt="csr",basis=basis_1d,dtype=np.float64)
    no_checks = dict(check_symm=False, check_pcon=False, check_herm=False)
    H = hamiltonian(static,[],static_fmt="csr",basis=basis_1d,dtype=np.float64,**no_checks)
# diagonalise H
    sizeH = H.tocsr(time=0).shape[0]
    if sizeH > 100:
#        ene,vec = H.eigsh(time=0.0,which="SA",k=2)
        ene = H.eigsh(which="SA",k=20,return_eigenvectors=False); ene = np.sort(ene)
#        ene = H.eigsh(which="SA",k=1,return_eigenvectors=False)
    else:
        ene = H.eigvalsh()
    return ene

def main():
    ###### define model parameters ######
    Ns = [i for i in range(6,22,2)]
    J1 = 1.0
    J2 = 0.5
    twoSz = 0
    for N in Ns:
        ene = exact_diag(J1,J2,N,twoSz)
#        print(N,ene[0],ene-ene[0])
        print(N,ene[0],' '.join(map(str,ene[0:20]-ene[0])))

if __name__ == "__main__":
    main()
