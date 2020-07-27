from __future__ import print_function, division
import sys,os

## http://weinbe58.github.io/QuSpin/generated/quspin.basis.spin_basis_general.html#quspin.basis.spin_basis_general

from quspin.operators import hamiltonian # operators
from quspin.basis import spin_basis_general # spin basis constructor
import numpy as np # general math functions
import scipy.sparse
import scipy.sparse.linalg
import scipy as scipy
import argparse
#import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='2d TFIsing model (- J \sum \sigma^z_i \sigma^z_j - g \sum \sigma^x_i, gc = 3.04438) sudden quench')
    parser.add_argument('-Lx',metavar='Lx',dest='Lx',type=int,default=3,help='set Lx')
    parser.add_argument('-Ly',metavar='Ly',dest='Ly',type=int,default=3,help='set Ly')
    parser.add_argument('-Ji',metavar='Ji',dest='Ji',type=float,default=0.0,help='set Ji (default:Ji=0,gi=1,large TF limit)')
    parser.add_argument('-Jf',metavar='Jf',dest='Jf',type=float,default=1.0,help='set Jf')
    parser.add_argument('-gi',metavar='gi',dest='gi',type=float,default=1.0,help='set gi (default:Ji=0,gi=1,large TF limit)')
    parser.add_argument('-gf',metavar='gf',dest='gf',type=float,default=0.1,help='set gf')
    return parser.parse_args()

def main():
    args = parse_args()
    Lx = args.Lx
    Ly = args.Ly
    Ji = args.Ji
    Jf = args.Jf
    gi = args.gi
    gf = args.gf
    ###### define model parameters ######
    #Lx, Ly = 3, 3 # linear dimension of spin 1 2d lattice
    #Lx, Ly = 4, 4 # linear dimension of spin 1 2d lattice
    #Lx, Ly = 5, 5 # linear dimension of spin 1 2d lattice
    N_2d = Lx*Ly # number of sites for spin 1
    J0 = -Ji
    g0 = -gi
    J = -Jf
    g = -gf
    ###### setting up user-defined symmetry transformations for 2d lattice ######
    s = np.arange(N_2d) # sites [0,1,2,....]
    x = s%Lx # x positions for sites
    y = s//Lx # y positions for sites
    T_x = (x+1)%Lx + Lx*y # translation along x-direction
    T_y = x +Lx*((y+1)%Ly) # translation along y-direction
    mT_x = (x+Lx-1)%Lx + Lx*y # translation along x-direction
    mT_y = x +Lx*((y+Ly-1)%Ly) # translation along y-direction
    P_x = x + Lx*(Ly-y-1) # reflection about x-axis
    P_y = (Lx-x-1) + Lx*y # reflection about y-axis
    Z   = -(s+1) # spin inversion
    ###### setting up bases ######
    basis_2d = spin_basis_general(N_2d,kxblock=(T_x,0),kyblock=(T_y,0),pxblock=(P_x,0),pyblock=(P_y,0),zblock=(Z,0))
    #basis0_2d = spin_basis_general(N_2d,Nup=N_2d)
    ###### setting up hamiltonian ######
    ### setting up site-coupling lists
    Jzz0 = [[J0,i,T_x[i]] for i in range(N_2d)]+[[J0,i,T_y[i]] for i in range(N_2d)]
    gx0 = [[g0,i] for i in range(N_2d)]
    Jzz = [[J,i,T_x[i]] for i in range(N_2d)]+[[J,i,T_y[i]] for i in range(N_2d)]
    gx = [[g,i] for i in range(N_2d)]
    int_m0 = [[1.0,i] for i in range(N_2d)]
    int_m00m01 = [[1.0,i,T_x[i]] for i in range(N_2d)]+[[1.0,i,T_y[i]] for i in range(N_2d)]
    int_m00m02 = [[1.0,i,T_x[T_x[i]]] for i in range(N_2d)]+[[1.0,i,T_y[T_y[i]]] for i in range(N_2d)]
    int_m00m11 = [[1.0,i,T_y[T_x[i]]] for i in range(N_2d)] + [[1.0,i,mT_y[T_x[i]]] for i in range(N_2d)]                                                                                                                                                                                                                         
    int_m00m22 = [[1.0,i,T_y[T_y[T_x[T_x[i]]]]] for i in range(N_2d)] + [[1.0,i,mT_y[mT_y[T_x[T_x[i]]]]] for i in range(N_2d)]                                                                                                                                                                                                                         
    static0 = [["zz",Jzz0],["x",gx0]]
    static = [["zz",Jzz],["x",gx]]
    static_mx = [["x",int_m0]]
    static_mz = [["z",int_m0]]
    static_mx00mx01 = [["xx",int_m00m01]]
    static_mz00mz01 = [["zz",int_m00m01]]
    static_mx00mx02 = [["xx",int_m00m02]]
    static_mz00mz02 = [["zz",int_m00m02]]
    static_mx00mx11 = [["xx",int_m00m11]]
    static_mz00mz11 = [["zz",int_m00m11]]
    static_mx00mx22 = [["xx",int_m00m22]]
    static_mz00mz22 = [["zz",int_m00m22]]
    #static0 = [["z",gz]]
    ### build hamiltonian
    no_checks=dict(check_symm=False, check_pcon=False, check_herm=False)
    H0 = hamiltonian(static0,[],static_fmt="csr",basis=basis_2d,dtype=np.float64,**no_checks).tocsr(time=0)
    H = hamiltonian(static,[],static_fmt="csr",basis=basis_2d,dtype=np.float64,**no_checks).tocsr(time=0)
    op_mx = hamiltonian(static_mx,[],static_fmt="csr",basis=basis_2d,dtype=np.float64,**no_checks).tocsr(time=0)
    op_mz = hamiltonian(static_mz,[],static_fmt="csr",basis=basis_2d,dtype=np.float64,**no_checks).tocsr(time=0)
    op_mx00mx01 = hamiltonian(static_mx00mx01,[],static_fmt="csr",basis=basis_2d,dtype=np.float64,**no_checks).tocsr(time=0)
    op_mz00mz01 = hamiltonian(static_mz00mz01,[],static_fmt="csr",basis=basis_2d,dtype=np.float64,**no_checks).tocsr(time=0)
    op_mx00mx02 = hamiltonian(static_mx00mx02,[],static_fmt="csr",basis=basis_2d,dtype=np.float64,**no_checks).tocsr(time=0)
    op_mz00mz02 = hamiltonian(static_mz00mz02,[],static_fmt="csr",basis=basis_2d,dtype=np.float64,**no_checks).tocsr(time=0)
    op_mx00mx11 = hamiltonian(static_mx00mx11,[],static_fmt="csr",basis=basis_2d,dtype=np.float64,**no_checks).tocsr(time=0)
    op_mz00mz11 = hamiltonian(static_mz00mz11,[],static_fmt="csr",basis=basis_2d,dtype=np.float64,**no_checks).tocsr(time=0)
    op_mx00mx22 = hamiltonian(static_mx00mx22,[],static_fmt="csr",basis=basis_2d,dtype=np.float64,**no_checks).tocsr(time=0)
    op_mz00mz22 = hamiltonian(static_mz00mz22,[],static_fmt="csr",basis=basis_2d,dtype=np.float64,**no_checks).tocsr(time=0)
    #print(H)
    #print(H0)
    ### diagonalize H
    ene0,vec0 = scipy.sparse.linalg.eigsh(H0,which='SA',k=2)
    print("Ji gi ene/N_2d",J0,g0,ene0/N_2d)
    #print(vec0)

    timei = 0.0
    timef = 5.0
    dt = 0.01
    Nsteps = int(timef/dt+0.1)+1
    list_time = [timei+i*(timef-timei)/(Nsteps-1) for i in range(Nsteps)]
    print("timei",timei)
    print("timef",timef)
    print("Nsteps",Nsteps)

    list_norm2 = []
    list_mx = []
    list_mz = []
    list_mx00mx01 = []
    list_mz00mz01 = []
    list_mx00mx02 = []
    list_mz00mz02 = []
    list_mx00mx11 = []
    list_mz00mz11 = []
    list_mx00mx22 = []
    list_mz00mz22 = []
    ret = vec0[:,0]

    dt = list_time[0]
    ret = (scipy.sparse.linalg.expm_multiply((-1j)*dt*H,ret,start=0.0,stop=1.0,num=2,endpoint=True))[1]
    norm2 = np.linalg.norm(ret)**2
    mx = (np.conjugate(ret).dot(op_mx.dot(ret)) / norm2).real / N_2d
    mz = (np.conjugate(ret).dot(op_mz.dot(ret)) / norm2).real / N_2d
    mx00mx01 = (np.conjugate(ret).dot(op_mx00mx01.dot(ret)) / norm2).real / N_2d
    mz00mz01 = (np.conjugate(ret).dot(op_mz00mz01.dot(ret)) / norm2).real / N_2d
    mx00mx02 = (np.conjugate(ret).dot(op_mx00mx02.dot(ret)) / norm2).real / N_2d
    mz00mz02 = (np.conjugate(ret).dot(op_mz00mz02.dot(ret)) / norm2).real / N_2d
    mx00mx11 = (np.conjugate(ret).dot(op_mx00mx11.dot(ret)) / norm2).real / N_2d
    mz00mz11 = (np.conjugate(ret).dot(op_mz00mz11.dot(ret)) / norm2).real / N_2d
    mx00mx22 = (np.conjugate(ret).dot(op_mx00mx22.dot(ret)) / norm2).real / N_2d
    mz00mz22 = (np.conjugate(ret).dot(op_mz00mz22.dot(ret)) / norm2).real / N_2d
    list_norm2.append(norm2)
    list_mx.append(mx)
    list_mz.append(mz)
    list_mx00mx01.append(mx00mx01)
    list_mz00mz01.append(mz00mz01)
    list_mx00mx02.append(mx00mx02)
    list_mz00mz02.append(mz00mz02)
    list_mx00mx11.append(mx00mx11)
    list_mz00mz11.append(mz00mz11)
    list_mx00mx22.append(mx00mx22)
    list_mz00mz22.append(mz00mz22)

    for i in range(1,Nsteps):
        dt = list_time[i] - list_time[i-1]
        ret = (scipy.sparse.linalg.expm_multiply((-1j)*dt*H,ret,start=0.0,stop=1.0,num=2,endpoint=True))[1]
        norm2 = np.linalg.norm(ret)**2
        mx = (np.conjugate(ret).dot(op_mx.dot(ret)) / norm2).real / N_2d
        mz = (np.conjugate(ret).dot(op_mz.dot(ret)) / norm2).real / N_2d
        mx00mx01 = (np.conjugate(ret).dot(op_mx00mx01.dot(ret)) / norm2).real / N_2d
        mz00mz01 = (np.conjugate(ret).dot(op_mz00mz01.dot(ret)) / norm2).real / N_2d
        mx00mx02 = (np.conjugate(ret).dot(op_mx00mx02.dot(ret)) / norm2).real / N_2d
        mz00mz02 = (np.conjugate(ret).dot(op_mz00mz02.dot(ret)) / norm2).real / N_2d
        mx00mx11 = (np.conjugate(ret).dot(op_mx00mx11.dot(ret)) / norm2).real / N_2d
        mz00mz11 = (np.conjugate(ret).dot(op_mz00mz11.dot(ret)) / norm2).real / N_2d
        mx00mx22 = (np.conjugate(ret).dot(op_mx00mx22.dot(ret)) / norm2).real / N_2d
        mz00mz22 = (np.conjugate(ret).dot(op_mz00mz22.dot(ret)) / norm2).real / N_2d
        list_norm2.append(norm2)
        list_mx.append(mx)
        list_mz.append(mz)
        list_mx00mx01.append(mx00mx01)
        list_mz00mz01.append(mz00mz01)
        list_mx00mx02.append(mx00mx02)
        list_mz00mz02.append(mz00mz02)
        list_mx00mx11.append(mx00mx11)
        list_mz00mz11.append(mz00mz11)
        list_mx00mx22.append(mx00mx22)
        list_mz00mz22.append(mz00mz22)
    print("list_time",list_time)
    print("list_norm2",list_norm2)
    print("list_mx",list_mx)
    print("list_mz",list_mz)
    print("list_mx00mx01",list_mx00mx01)
    print("list_mz00mz01",list_mz00mz01)
    print("list_mx00mx02",list_mx00mx02)
    print("list_mz00mz02",list_mz00mz02)
    print("list_mx00mx11",list_mx00mx11)
    print("list_mz00mz11",list_mz00mz11)
    print("list_mx00mx22",list_mx00mx22)
    print("list_mz00mz22",list_mz00mz22)

    fig = plt.figure()
    fig.suptitle("|norm|$^2$")
    plt.plot(list_time,list_norm2)
    plt.xlabel("$t$")
    fig.savefig("fig_norm2.png")

    fig = plt.figure()
    fig.suptitle("$m_x$")
    plt.plot(list_time,list_mx)
    plt.xlabel("$t$")
    fig.savefig("fig_mx.png")

    fig = plt.figure()
    fig.suptitle("$m_z$")
    plt.plot(list_time,list_mz)
    plt.xlabel("$t$")
    fig.savefig("fig_mz.png")

    fig = plt.figure()
    fig.suptitle("$m_x(0,0)m_x(0,1)$")
    plt.plot(list_time,list_mx00mx01)
    plt.xlabel("$t$")
    fig.savefig("fig_mx00mx01.png")

    fig = plt.figure()
    fig.suptitle("$m_z(0,0)m_z(0,1)$")
    plt.plot(list_time,list_mz00mz01)
    plt.xlabel("$t$")
    fig.savefig("fig_mz00mz01.png")

    fig = plt.figure()
    fig.suptitle("$m_x(0,0)m_x(0,2)$")
    plt.plot(list_time,list_mx00mx02)
    plt.xlabel("$t$")
    fig.savefig("fig_mx00mx02.png")

    fig = plt.figure()
    fig.suptitle("$m_z(0,0)m_z(0,2)$")
    plt.plot(list_time,list_mz00mz02)
    plt.xlabel("$t$")
    fig.savefig("fig_mz00mz02.png")

    fig = plt.figure()
    fig.suptitle("$m_x(0,0)m_x(1,1)$")
    plt.plot(list_time,list_mx00mx11)
    plt.xlabel("$t$")
    fig.savefig("fig_mx00mx11.png")

    fig = plt.figure()
    fig.suptitle("$m_z(0,0)m_z(1,1)$")
    plt.plot(list_time,list_mz00mz11)
    plt.xlabel("$t$")
    fig.savefig("fig_mz00mz11.png")

    fig = plt.figure()
    fig.suptitle("$m_x(0,0)m_x(2,2)$")
    plt.plot(list_time,list_mx00mx22)
    plt.xlabel("$t$")
    fig.savefig("fig_mx00mx22.png")

    fig = plt.figure()
    fig.suptitle("$m_z(0,0)m_z(2,2)$")
    plt.plot(list_time,list_mz00mz22)
    plt.xlabel("$t$")
    fig.savefig("fig_mz00mz22.png")

    plt.close()
    #plt.show()

if __name__ == "__main__":                                                                                                                                                                                                                                                                                                    
    main()
