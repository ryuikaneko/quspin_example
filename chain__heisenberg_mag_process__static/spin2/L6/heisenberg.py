#!/usr/bin/env python

from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def exact_diag(J1,J2,L,m,k,p):
    N = L # number of sites
###### setting up bases ######
## https://github.com/weinbe58/QuSpin/issues/324
#    basis_1d = spin_basis_1d(L=N,m=m,S="2",pauli=0)
#    basis_1d = spin_basis_1d(L=N,m=m,S="2",pauli=0,a=1,kblock=k)
    basis_1d = spin_basis_1d(L=N,m=m,S="2",pauli=0,kblock=k,pblock=p)
#    basis_1d = spin_basis_1d(L=N,m=m,S="2",pauli=0,kblock=k,pblock=p)
#    basis_1d = spin_basis_1d(L=N,m=m,S="2",pauli=0,a=1,kblock=k,pblock=p)
#    basis_1d = spin_basis_1d(L=N,m=0,S="2",pauli=0,a=1,kblock=0,pblock=1,zblock=1)
###### setting up hamiltonian ######
    Jzzs = [] \
      + [[J1,i,(i+1)%N] for i in range(N)] \
      + [[J2,i,(i+2)%N] for i in range(N)]
    Jpms = [] \
      + [[0.5*J1,i,(i+1)%N] for i in range(N)] \
      + [[0.5*J2,i,(i+2)%N] for i in range(N)]
    Jmps = [] \
      + [[0.5*J1,i,(i+1)%N] for i in range(N)] \
      + [[0.5*J2,i,(i+2)%N] for i in range(N)]
    static = [\
        ["zz",Jzzs],["+-",Jpms],["-+",Jmps],\
        ]
# build hamiltonian
#    H = hamiltonian(static,[],static_fmt="csr",basis=basis_1d,dtype=np.float64)
    no_checks = dict(check_symm=False,check_pcon=False,check_herm=False)
    H = hamiltonian(static,[],static_fmt="csr",basis=basis_1d,dtype=np.float64,**no_checks)
# diagonalise H
    sizeH = H.tocsr(time=0).shape[0]
    if sizeH > 100:
#        ene,vec = H.eigsh(time=0.0,which="SA",k=2)
#        ene = H.eigsh(which="SA",k=2,return_eigenvectors=False); ene = np.sort(ene)
        ene = H.eigsh(which="SA",k=1,return_eigenvectors=False)
    else:
        ene = H.eigvalsh()
    return ene

def main():
    ###### define model parameters ######
#    L = 4 # length of chain
    L = 6 # length of chain
    J1 = 1.0 # spin-spin interaction
    J2 = 0.0 # spin-spin interaction
    spin = 2
    spinL = int(spin*L + 0.01)

    intms = [i for i in range(spinL+1)]
    list_intm = []
    list_ene = []
    for intm in intms:
        m = intm * 1.0/L
#----
## search all momenta
        list_tmp_ene = []
        print("## intm",intm)
        if intm == spinL: ## all up
            k = 0
            p = +1
            ene = exact_diag(J1,J2,L,m,k,p)
            if ene.size: # if ene is not empty
                list_tmp_ene.append(ene[0])
                print("#",spin*L,J1,J2,intm,m,k,p,ene[0])
        else:
            for k in range(L//2+1):
                for p in [+1,-1]:
                    ene = exact_diag(J1,J2,L,m,k,p)
                    if ene.size: # if ene is not empty
                        list_tmp_ene.append(ene[0])
                        print("#",spin*L,J1,J2,intm,m,k,p,ene[0])
#----
        ene = min(list_tmp_ene)
        print(spin*L,J1,J2,intm,m,ene)
        list_intm.append(intm)
        list_ene.append(ene)

## calculate magnetization process
    hmax = list_ene[-1] - list_ene[-2]
    list_h = np.linspace(0,hmax*1.05,1051)
    list_magproc = np.array(
        [
            np.argmin(np.array(
                [list_ene[i]-list_intm[i]*h for i in range(spinL+1)]
            ))
            for h in list_h
        ]
        )/L
#    print(hmax)
#    print(list_intm)
#    print(list_ene)
#    print(list_h)
#    print(list_magproc)

    np.savetxt("dat_1_intm",np.array(list_intm))
    np.savetxt("dat_1_ene",np.array(list_ene))
    np.savetxt("dat_2_h",list_h)
    np.savetxt("dat_2_magproc",list_magproc)

    fig = plt.figure()
    plt.plot(list_h,list_magproc)
    plt.xlabel("H")
    plt.ylabel("M")
    fig.savefig("fig_magproc.png")
    plt.close()

if __name__ == "__main__":
    main()
