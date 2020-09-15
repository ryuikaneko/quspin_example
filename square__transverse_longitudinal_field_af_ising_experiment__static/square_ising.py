## http://weinbe58.github.io/QuSpin/generated/quspin.basis.spin_basis_general.html#quspin.basis.spin_basis_general
## https://doi.org/10.1103/PhysRevX.8.021069
## https://doi.org/10.1103/PhysRevX.8.021070
## https://doi.org/10.1038/nature24622
## consider nearest neighbor Ising
from __future__ import print_function, division
from quspin.operators import hamiltonian # operators
from quspin.basis import spin_basis_general # spin basis constructor
import numpy as np # general math functions

def exact_diag(J,Hx,Hz,Lx,Ly):
    N_2d = Lx*Ly # number of sites
    ###### setting up user-defined symmetry transformations for 2d lattice ######
    s = np.arange(N_2d) # sites [0,1,2,....]
    x = s%Lx # x positions for sites
    y = s//Lx # y positions for sites
    T_x = (x+1)%Lx + Lx*y # translation along x-direction
    T_y = x +Lx*((y+1)%Ly) # translation along y-direction
    mT_y = x +Lx*((y+Ly-1)%Ly) # translation along y-direction
    P_x = x + Lx*(Ly-y-1) # reflection about x-axis
    P_y = (Lx-x-1) + Lx*y # reflection about y-axis
    Z   = -(s+1) # spin inversion
    ###### setting up bases ######
#    basis_2d = spin_basis_general(N=N_2d,S="1/2",pauli=0)
    basis_2d = spin_basis_general(N=N_2d,S="1/2",pauli=0,kxblock=(T_x,0),kyblock=(T_y,0))
    ###### setting up hamiltonian ######
    # setting up site-coupling lists
    Jzzs = [[J,i,T_x[i]] for i in range(N_2d)]+[[J,i,T_y[i]] for i in range(N_2d)]
    Hxs = [[-Hx,i] for i in range(N_2d)]
    Hzs = [[-Hz,i] for i in range(N_2d)]
    static = [["zz",Jzzs],["x",Hxs],["z",Hzs]]
    # build hamiltonian
#    H = hamiltonian(static,[],static_fmt="csr",basis=basis_2d,dtype=np.float64)
    no_checks = dict(check_symm=False, check_pcon=False, check_herm=False)
    H = hamiltonian(static,[],static_fmt="csr",basis=basis_2d,dtype=np.float64,**no_checks)
    # diagonalise H
    ene,vec = H.eigsh(time=0.0,which="SA",k=2)
#    ene = H.eigsh(time=0.0,which="SA",k=2,return_eigenvectors=False); ene = np.sort(ene)
    norm2 = np.linalg.norm(vec[:,0])**2
    # calculate uniform magnetization
    int_mx = [[1.0,i] for i in range(N_2d)]
    int_mz = [[1.0,i] for i in range(N_2d)]
    static_mx = [["x",int_mx]]
    static_mz = [["z",int_mz]]
    op_mx = hamiltonian(static_mx,[],static_fmt="csr",basis=basis_2d,dtype=np.float64,**no_checks).tocsr(time=0)
    op_mz = hamiltonian(static_mz,[],static_fmt="csr",basis=basis_2d,dtype=np.float64,**no_checks).tocsr(time=0)
    mx = (np.conjugate(vec[:,0]).dot(op_mx.dot(vec[:,0])) / norm2).real / N_2d
    mz = (np.conjugate(vec[:,0]).dot(op_mz.dot(vec[:,0])) / norm2).real / N_2d
    # calculate n.n. sz.sz correlation
    int_mz0mz1 = [[1.0,i,T_x[i]] for i in range(N_2d)]+[[1.0,i,T_y[i]] for i in range(N_2d)]
    static_mz0mz1 = [["zz",int_mz0mz1]]
    op_mz0mz1 = hamiltonian(static_mz0mz1,[],static_fmt="csr",basis=basis_2d,dtype=np.float64,**no_checks).tocsr(time=0)
    mz0mz1 = (np.conjugate(vec[:,0]).dot(op_mz0mz1.dot(vec[:,0])) / norm2).real / N_2d
    # calculate sz(0,0).sz(1,1) correlation
    int_mz0mzsq2 = [[1.0,i,T_y[T_x[i]]] for i in range(N_2d)]+[[1.0,i,mT_y[T_x[i]]] for i in range(N_2d)]
    static_mz0mzsq2 = [["zz",int_mz0mzsq2]]
    op_mz0mzsq2 = hamiltonian(static_mz0mzsq2,[],static_fmt="csr",basis=basis_2d,dtype=np.float64,**no_checks).tocsr(time=0)
    mz0mzsq2 = (np.conjugate(vec[:,0]).dot(op_mz0mzsq2.dot(vec[:,0])) / norm2).real / N_2d
    return ene, mx, mz, mz0mz1, mz0mzsq2

def main():
###### define model parameters ######
    Dim = 2
    Lx, Ly = 4, 4 # linear dimension of 2d lattice
    N_2d = Lx*Ly # number of sites
#
## H = + Omega \sum_i S_i^x
##     - (Delta - V * Dim) \sum_{i} S_i^z
##     + V \sum_{<ij>} S_i^z S_j^z
##     + (V * Dim / 4 - Delta / 2) * \sum_{i} 1_i
#
    npointsO = 21
    npointsD = 51
    Omegas = np.linspace(0.0,2.0,npointsO) # Rabi frequency
    Deltas = np.linspace(-0.5,4.5,npointsD) # detuning
    V = 1.0 # van der Waals
#
    print("#V,Omega,Delta,J,Hz,Hx,Lx,Ly,ene[0]/N_2d+eshift,mx,mz,mz0mz1,mz0mz1-mz**2,mz0mzsq2,mz0mzsq2-mz**2")
    for Omega in Omegas:
        for Delta in Deltas:
            Hx = - Omega # transverse field
            Hz = Delta - V*Dim # longitudinal field
            J = V # AF Ising
            eshift = 0.25*V*Dim - 0.5*Delta
            ene, mx, mz, mz0mz1, mz0mzsq2 = exact_diag(J,Hx,Hz,Lx,Ly)
            print(V,Omega,Delta,J,Hz,Hx,Lx,Ly,ene[0]/N_2d+eshift,mx,mz,mz0mz1,mz0mz1-mz**2,mz0mzsq2,mz0mzsq2-mz**2)
        print()

if __name__ == "__main__":
    main()
