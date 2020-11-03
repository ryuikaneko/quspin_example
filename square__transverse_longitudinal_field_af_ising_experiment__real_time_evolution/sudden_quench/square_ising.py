## http://weinbe58.github.io/QuSpin/generated/quspin.basis.spin_basis_general.html#quspin.basis.spin_basis_general
## https://doi.org/10.1103/PhysRevX.8.021069
## https://doi.org/10.1103/PhysRevX.8.021070
## https://doi.org/10.1038/nature24622
## consider nearest neighbor Ising
from __future__ import print_function, division
from quspin.operators import hamiltonian # operators
from quspin.basis import spin_basis_general # spin basis constructor
import numpy as np # general math functions
import scipy.sparse.linalg
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="transverse longitudinal field Ising model on a square lattice, dynamics")
    parser.add_argument("-Lx",metavar="Lx",dest="Lx",type=int,default=4,help="set Lx")
    parser.add_argument("-Ly",metavar="Ly",dest="Ly",type=int,default=4,help="set Ly")
    parser.add_argument("-npoints",metavar="npoints",dest="npoints",type=int,default=1001,help="set npoints")
    parser.add_argument("-V",metavar="V",dest="V",type=np.float,default=1.0,help="set V")
    parser.add_argument("-Omega",metavar="Omega",dest="Omega",type=np.float,default=1.4,help="set Omega")
    parser.add_argument("-Delta",metavar="Delta",dest="Delta",type=np.float,default=2.0,help="set Delta")
    parser.add_argument("-Tau",metavar="Tau",dest="Tau",type=np.float,default=10.0,help="set Tau")
    return parser.parse_args()

def prepare_H_vec(J,Hx,Hz,Lx,Ly):
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
#    print(basis_2d)
#    print(basis_2d.Ns)
    ###### prepare initial state (all down, fock state |000...000> in QuSpin) ######
## http://weinbe58.github.io/QuSpin/generated/quspin.basis.spinful_fermion_basis_1d.html?highlight=partial%20trace#quspin.basis.spinful_fermion_basis_1d.index
## http://weinbe58.github.io/QuSpin/generated/quspin.basis.spin_basis_1d.html#quspin.basis.spin_basis_1d.index
##    i0 = basis_2d.index("1111111111111111") # up
#    i0 = basis_2d.index("0000000000000000") # down
    s_down = "".join("0" for i in range(N_2d))
    i_down = basis_2d.index(s_down)
    vec = np.zeros(basis_2d.Ns,dtype=np.float64)
    vec[i_down] = 1.0
#    print(s_down)
#    print(i_down)
#    print(vec)
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
    H = H.tocsr(time=0)
    # operator for uniform magnetization
    int_mx = [[1.0,i] for i in range(N_2d)]
    int_mz = [[1.0,i] for i in range(N_2d)]
    static_mx = [["x",int_mx]]
    static_mz = [["z",int_mz]]
    op_mx = hamiltonian(static_mx,[],static_fmt="csr",basis=basis_2d,dtype=np.float64,**no_checks).tocsr(time=0)
    op_mz = hamiltonian(static_mz,[],static_fmt="csr",basis=basis_2d,dtype=np.float64,**no_checks).tocsr(time=0)
    # operator for n.n. sz.sz correlation
    int_mz0mz1 = [[1.0,i,T_x[i]] for i in range(N_2d)]+[[1.0,i,T_y[i]] for i in range(N_2d)]
    static_mz0mz1 = [["zz",int_mz0mz1]]
    op_mz0mz1 = hamiltonian(static_mz0mz1,[],static_fmt="csr",basis=basis_2d,dtype=np.float64,**no_checks).tocsr(time=0)
    # operator for sz(0,0).sz(1,1) correlation
    int_mz0mzsq2 = [[1.0,i,T_y[T_x[i]]] for i in range(N_2d)]+[[1.0,i,mT_y[T_x[i]]] for i in range(N_2d)]
    static_mz0mzsq2 = [["zz",int_mz0mzsq2]]
    op_mz0mzsq2 = hamiltonian(static_mz0mzsq2,[],static_fmt="csr",basis=basis_2d,dtype=np.float64,**no_checks).tocsr(time=0)
    # operator for sz(0,0).sz(0,2) correlation
    int_mz0mz2 = [[1.0,i,T_x[T_x[i]]] for i in range(N_2d)]+[[1.0,i,T_y[T_y[i]]] for i in range(N_2d)]
    static_mz0mz2 = [["zz",int_mz0mz2]]
    op_mz0mz2 = hamiltonian(static_mz0mz2,[],static_fmt="csr",basis=basis_2d,dtype=np.float64,**no_checks).tocsr(time=0)
    return N_2d, H, vec, op_mx, op_mz, op_mz0mz1, op_mz0mzsq2, op_mz0mz2

def apply_expm(dt,N_2d,H,vec,op_mx,op_mz,op_mz0mz1,op_mz0mzsq2,op_mz0mz2):
    vec2 = (scipy.sparse.linalg.expm_multiply((-1j)*dt*H,vec,start=0.0,stop=1.0,num=2,endpoint=True))[1]
    norm2 = np.linalg.norm(vec2)**2
    # calculate energy
    ene = (np.conjugate(vec2).dot(H.dot(vec2)) / norm2).real / N_2d
    # calculate uniform magnetization
    mx = (np.conjugate(vec2).dot(op_mx.dot(vec2)) / norm2).real / N_2d
    mz = (np.conjugate(vec2).dot(op_mz.dot(vec2)) / norm2).real / N_2d
    # calculate n.n. sz.sz correlation
    mz0mz1 = (np.conjugate(vec2).dot(op_mz0mz1.dot(vec2)) / norm2).real / N_2d
    # calculate sz(0,0).sz(1,1) correlation
    mz0mzsq2 = (np.conjugate(vec2).dot(op_mz0mzsq2.dot(vec2)) / norm2).real / N_2d
    # calculate sz(0,0).sz(0,2) correlation
    mz0mz2 = (np.conjugate(vec2).dot(op_mz0mz2.dot(vec2)) / norm2).real / N_2d
    return vec2, norm2, ene, mx, mz, mz0mz1, mz0mzsq2, mz0mz2

def main():
    args = parse_args()
###### define model parameters ######
    Dim = 2
#    Lx, Ly = 4, 4 # linear dimension of 2d lattice
    Lx, Ly = args.Lx, args.Ly # linear dimension of 2d lattice
    N_2d = Lx*Ly # number of sites

## H = + Omega \sum_i S_i^x
##     - (Delta - V * Dim) \sum_{i} S_i^z
##     + V \sum_{<ij>} S_i^z S_j^z
##     + (V * Dim / 4 - Delta / 2) * \sum_{i} 1_i

    print("#Time,V,Omega,Delta,J,Hz,Hx,Lx,Ly,ene+eshift,norm2,mx,mz,mz0mz1/2,mz0mz1/2-mz**2,mz0mzsq2/2,mz0mzsq2/2-mz**2,mz0mz2/2,mz0mz2/2-mz**2")

## at time=0
#    V = 1.0
#    Omega = 1.4
#    Delta = 2.0
    V = args.V
    Omega = args.Omega
    Delta = args.Delta
#
    J = V
    Hx = - Omega
    Hz = Delta - V*Dim
    eshift = 0.25*V*Dim - 0.5*Delta
#
    Time = 0.0
    dt = 0.0
#
    N_2d, H, vec, op_mx, op_mz, op_mz0mz1, op_mz0mzsq2, op_mz0mz2 = prepare_H_vec(J,Hx,Hz,Lx,Ly)
    vec, norm2, ene, mx, mz, mz0mz1, mz0mzsq2, mz0mz2 = apply_expm(dt,N_2d,H,vec,op_mx,op_mz,op_mz0mz1,op_mz0mzsq2,op_mz0mz2)
    print(Time,V,Omega,Delta,J,Hz,Hx,Lx,Ly,ene+eshift,norm2,mx,mz,mz0mz1/2,mz0mz1/2-mz**2,mz0mzsq2/2,mz0mzsq2/2-mz**2,mz0mz2/2,mz0mz2/2-mz**2)
#    print(vec)

## real time evolution
#    npoints = 1001
#    Tau = 10.0
    npoints = args.npoints
    Tau = args.Tau
    Times = np.linspace(0.0,Tau,npoints)
#
    for i in range(1,npoints):
        Time = Times[i]
        dt = Times[i]-Times[i-1]
        vec, norm2, ene, mx, mz, mz0mz1, mz0mzsq2, mz0mz2 = apply_expm(dt,N_2d,H,vec,op_mx,op_mz,op_mz0mz1,op_mz0mzsq2,op_mz0mz2)
        print(Time,V,Omega,Delta,J,Hz,Hx,Lx,Ly,ene+eshift,norm2,mx,mz,mz0mz1/2,mz0mz1/2-mz**2,mz0mzsq2/2,mz0mzsq2/2-mz**2,mz0mz2/2,mz0mz2/2-mz**2)

if __name__ == "__main__":
    main()
