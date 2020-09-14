from __future__ import print_function, division
from quspin.operators import hamiltonian # operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
import numpy as np # general math functions

def exact_diag(Jleg,Jrung,J4,Dleg,Drung,L,pb,zb):
    N = 2*L # number of sites
###### setting up bases ######
#    basis_1d = spin_basis_1d(L=N,Nup=N//2,S="1/2",pauli=0)
    basis_1d = spin_basis_1d(L=N,Nup=N//2,S="1/2",pauli=0,a=2,pblock=pb,zblock=zb)
###### setting up hamiltonian ######
    Jzzs = \
        [[Jleg*Dleg,i,(i+2)%N] for i in range(0,N,2)] \
      + [[Jleg*Dleg,i,(i+2)%N] for i in range(1,N,2)] \
      + [[Jrung*Drung,i,i+1] for i in range(0,N,2)]
#    Jpms = \
#        [[0.5*Jleg,i,(i+2)%N] for i in range(0,N,2)] \
#      + [[0.5*Jleg,i,(i+2)%N] for i in range(1,N,2)] \
#      + [[0.5*Jrung,i,i+1] for i in range(0,N,2)]
    Jpms = \
        [[ 0.5*Jleg,i,(i+2)%N] for i in range(0,N-2,2)] \
      + [[ 0.5*Jleg,i,(i+2)%N] for i in range(1,N-2,2)] \
      + [[-0.5*Jleg,i,(i+2)%N] for i in range(N-2,N,2)] \
      + [[-0.5*Jleg,i,(i+2)%N] for i in range(N-1,N,2)] \
      + [[ 0.5*Jrung,i,i+1] for i in range(0,N,2)]
#    Jmps = \
#        [[0.5*Jleg,i,(i+2)%N] for i in range(0,N,2)] \
#      + [[0.5*Jleg,i,(i+2)%N] for i in range(1,N,2)] \
#      + [[0.5*Jrung,i,i+1] for i in range(0,N,2)]
    Jmps = \
        [[ 0.5*Jleg,i,(i+2)%N] for i in range(0,N-2,2)] \
      + [[ 0.5*Jleg,i,(i+2)%N] for i in range(1,N-2,2)] \
      + [[-0.5*Jleg,i,(i+2)%N] for i in range(N-2,N,2)] \
      + [[-0.5*Jleg,i,(i+2)%N] for i in range(N-1,N,2)] \
      + [[ 0.5*Jrung,i,i+1] for i in range(0,N,2)]
    Jzzzzs = [[     J4,i,(i+2)%N,(i+1)%N,(i+3)%N] for i in range(0,N,2)]
#    Jzzpms = [[ 0.5*J4,i,(i+2)%N,(i+1)%N,(i+3)%N] for i in range(0,N,2)]
    Jzzpms = \
        [[  0.5*J4,i,(i+2)%N,(i+1)%N,(i+3)%N] for i in range(0,N-2,2)] \
      + [[ -0.5*J4,i,(i+2)%N,(i+1)%N,(i+3)%N] for i in range(N-2,N,2)]
#    Jzzmps = [[ 0.5*J4,i,(i+2)%N,(i+1)%N,(i+3)%N] for i in range(0,N,2)]
    Jzzmps = \
        [[  0.5*J4,i,(i+2)%N,(i+1)%N,(i+3)%N] for i in range(0,N-2,2)] \
      + [[ -0.5*J4,i,(i+2)%N,(i+1)%N,(i+3)%N] for i in range(N-2,N,2)]
#    Jpmzzs = [[ 0.5*J4,i,(i+2)%N,(i+1)%N,(i+3)%N] for i in range(0,N,2)]
    Jpmzzs = \
        [[  0.5*J4,i,(i+2)%N,(i+1)%N,(i+3)%N] for i in range(0,N-2,2)] \
      + [[ -0.5*J4,i,(i+2)%N,(i+1)%N,(i+3)%N] for i in range(N-2,N,2)]
    Jpmpms = [[0.25*J4,i,(i+2)%N,(i+1)%N,(i+3)%N] for i in range(0,N,2)]
    Jpmmps = [[0.25*J4,i,(i+2)%N,(i+1)%N,(i+3)%N] for i in range(0,N,2)]
#    Jmpzzs = [[ 0.5*J4,i,(i+2)%N,(i+1)%N,(i+3)%N] for i in range(0,N,2)]
    Jmpzzs = \
        [[  0.5*J4,i,(i+2)%N,(i+1)%N,(i+3)%N] for i in range(0,N-2,2)] \
      + [[ -0.5*J4,i,(i+2)%N,(i+1)%N,(i+3)%N] for i in range(N-2,N,2)]
    Jmppms = [[0.25*J4,i,(i+2)%N,(i+1)%N,(i+3)%N] for i in range(0,N,2)]
    Jmpmps = [[0.25*J4,i,(i+2)%N,(i+1)%N,(i+3)%N] for i in range(0,N,2)]
    static = [\
        ["zz",Jzzs],["+-",Jpms],["-+",Jmps],\
        ["zzzz",Jzzzzs],["zz+-",Jzzpms],["zz-+",Jzzmps],\
        ["+-zz",Jpmzzs],["+-+-",Jpmpms],["+--+",Jpmmps],\
        ["-+zz",Jmpzzs],["-++-",Jmppms],["-+-+",Jmpmps],\
        ]
# build hamiltonian
#    H = hamiltonian(static,[],static_fmt="csr",basis=basis_1d,dtype=np.float64)
    no_checks = dict(check_symm=False, check_pcon=False, check_herm=False)
    H = hamiltonian(static,[],static_fmt="csr",basis=basis_1d,dtype=np.float64,**no_checks)
# diagonalise H
#    ene,vec = H.eigsh(time=0.0,which="SA",k=2)
    ene = H.eigsh(which="SA",k=2,return_eigenvectors=False); ene = np.sort(ene)
    return ene

def main():
    ## 2-leg ladder (L=inf): -0.578043140180 (PhysRevB.89.094424, see also PhysRevB.54.R3714, PhysRevB.47.3196)
    ## ladder + ring exchange: PhysRevB.80.014426, PhysRevB.88.104403
    ###### define model parameters ######
    Jleg = 1.0 # spin-spin interaction, leg
    Jrung = 1.0 # spin-spin interaction, rung
    L = 8 # length of chain
#    L = 14 # length of chain

    Dleg = 1.0 # Ising anisotropy, leg
    Drung = 1.0 # Ising anisotropy, rung
#    J4 = 0.0
#    list_J4 = [0.0]
#    list_J4 = [1.19]
    list_J4 = np.linspace(1.179,1.201,23)
    for J4 in list_J4:
        enepp = exact_diag(Jleg,Jrung,J4,Dleg,Drung,L,+1,+1)
        enepm = exact_diag(Jleg,Jrung,J4,Dleg,Drung,L,+1,-1)
        enemp = exact_diag(Jleg,Jrung,J4,Dleg,Drung,L,-1,+1)
        enemm = exact_diag(Jleg,Jrung,J4,Dleg,Drung,L,-1,-1)
        gap = enepp[0] - enemm[0] ## gap between 2 excited states
        print(2*L,Jleg,Jrung,J4,enepp[0],enepm[0],enemp[0],enemm[0],gap)

if __name__ == "__main__":
    main()
