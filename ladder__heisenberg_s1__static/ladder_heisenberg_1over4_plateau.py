# https://doi.org/10.1103/PhysRevB.74.024421
# https://doi.org/10.1103/PhysRevB.81.014407
# https://doi.org/10.1143/JPSJ.70.636 https://arxiv.org/abs/cond-mat/0011034
# https://doi.org/10.1143/JPSJS.74S.165

from __future__ import print_function, division
from quspin.operators import hamiltonian # operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
import numpy as np # general math functions
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#def exact_diag(Jleg1,Jleg2,Jrung,Dleg1,Dleg2,Drung,L,m):
#def exact_diag(Jleg1,Jleg2,Jrung,Dleg1,Dleg2,Drung,L,m,k):
def exact_diag(Jleg1,Jleg2,Jrung,Dleg1,Dleg2,Drung,L,m,k,p):
    N = 2*L # number of sites
###### setting up bases ######
## https://github.com/weinbe58/QuSpin/issues/324
#    basis_1d = spin_basis_1d(L=N,m=m,S="1",pauli=0)
#    basis_1d = spin_basis_1d(L=N,m=m,S="1",pauli=0,a=2,kblock=k)
    basis_1d = spin_basis_1d(L=N,m=m,S="1",pauli=0,a=2,kblock=k,pblock=p)
#    basis_1d = spin_basis_1d(L=N,m=0,S="1",pauli=0,a=2,kblock=0,pblock=1,zblock=1)
###### setting up hamiltonian ######
    Jzzs = \
        [[Jleg1*Dleg1,i,(i+2)%N] for i in range(0,N,2)] \
      + [[Jleg1*Dleg1,i,(i+2)%N] for i in range(1,N,2)] \
      + [[Jleg2*Dleg2,i,(i+3)%N] for i in range(0,N,2)] \
      + [[Jleg2*Dleg2,i,(i+1)%N] for i in range(1,N,2)] \
      + [[Jrung*Drung,i,i+1] for i in range(0,N,2)]
    Jpms = \
        [[0.5*Jleg1,i,(i+2)%N] for i in range(0,N,2)] \
      + [[0.5*Jleg1,i,(i+2)%N] for i in range(1,N,2)] \
      + [[0.5*Jleg2,i,(i+3)%N] for i in range(0,N,2)] \
      + [[0.5*Jleg2,i,(i+1)%N] for i in range(1,N,2)] \
      + [[0.5*Jrung,i,i+1] for i in range(0,N,2)]
    Jmps = \
        [[0.5*Jleg1,i,(i+2)%N] for i in range(0,N,2)] \
      + [[0.5*Jleg1,i,(i+2)%N] for i in range(1,N,2)] \
      + [[0.5*Jleg2,i,(i+3)%N] for i in range(0,N,2)] \
      + [[0.5*Jleg2,i,(i+1)%N] for i in range(1,N,2)] \
      + [[0.5*Jrung,i,i+1] for i in range(0,N,2)]
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
#        ene = H.eigsh(which="SA",k=2,return_eigenvectors=False); ene = np.sort(ene)
        ene = H.eigsh(which="SA",k=1,return_eigenvectors=False)
    else:
        ene = H.eigvalsh()
    return ene

def main():
    ###### define model parameters ######
#    L = 4 # length of chain
    L = 6 # length of chain
    Jleg1 = 0.5 # spin-spin interaction, leg1
    Jleg2 = 0.5 # spin-spin interaction, leg2
    Jrung = 1.0 # spin-spin interaction, rung
    Dleg1 = 1.0 # Ising anisotropy, leg1
    Dleg2 = 1.0 # Ising anisotropy, leg2
    Drung = 1.0 # Ising anisotropy, rung

    intms = [i for i in range(2*L+1)]
    list_intm = []
    list_ene = []
    for intm in intms:
        m = intm * 0.5/L
#----
## without momentum conservation
#        ene = exact_diag(Jleg1,Jleg2,Jrung,Dleg1,Dleg2,Drung,L,m)
#----
## guess momentum
#        if intm%2==0:
#            k = 0
#        else:
#            k = L//2
#        ene = exact_diag(Jleg1,Jleg2,Jrung,Dleg1,Dleg2,Drung,L,m,k)
#----
## search all momenta
        list_tmp_ene = []
        if intm == 2*L:
            k = 0
            p = +1
            ene = exact_diag(Jleg1,Jleg2,Jrung,Dleg1,Dleg2,Drung,L,m,k,p)
            list_tmp_ene.append(ene[0])
            print("#",2*L,Jleg1,Jrung,intm,m,k,p,ene[0])
        else:
            for k in range(L//2+1):
                for p in [+1,-1]:
                    ene = exact_diag(Jleg1,Jleg2,Jrung,Dleg1,Dleg2,Drung,L,m,k,p)
                    list_tmp_ene.append(ene[0])
                    print("#",2*L,Jleg1,Jrung,intm,m,k,p,ene[0])
#----
        ene = min(list_tmp_ene)
        print(2*L,Jleg1,Jrung,intm,m,ene)
        list_intm.append(intm)
        list_ene.append(ene)

## calculate magnetization process
    hmax = list_ene[-1] - list_ene[-2]
    list_h = np.linspace(0,hmax*1.05,1051)
    list_magproc = np.array(
        [
            np.argmin(np.array(
                [list_ene[i]-list_intm[i]*h for i in range(2*L+1)]
            ))
            for h in list_h
        ]
        )/(2*L)
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
