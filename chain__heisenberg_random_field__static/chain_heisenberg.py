# http://weinbe58.github.io/QuSpin/examples/example6.html
# https://doi.org/10.1103/PhysRevB.82.174411

from __future__ import print_function, division
from quspin.operators import hamiltonian,quantum_operator
from quspin.basis import spin_basis_1d
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="MBL")
    parser.add_argument("-N",metavar="N",dest="N",type=int,default=6,help="set N")
    parser.add_argument("-Nsmp",metavar="Nsmp",dest="Nsmp",type=int,default=128,help="set Nsmp")
    parser.add_argument("-Hz",metavar="Hz",dest="Hz",type=np.float64,default=0.0,help="set Hz")
    return parser.parse_args()

def make_operator(J,N,twoSz):
    basis_1d = spin_basis_1d(L=N,Nup=N//2+twoSz,S="1/2",pauli=0)
    Jzzs = [[J,i,(i+1)%N] for i in range(N)]
    Jpms = [[0.5*J,i,(i+1)%N] for i in range(N)]
    Jmps = [[0.5*J,i,(i+1)%N] for i in range(N)]
    static = [["zz",Jzzs],["+-",Jpms],["-+",Jmps]]
    operator_dict = dict(H0=static)
    for i in range(N):
        operator_dict["z"+str(i)] = [["z",[[1.0,i]]]]
    no_checks = dict(check_symm=False,check_pcon=False,check_herm=False)
    H_dict = quantum_operator(operator_dict,basis=basis_1d,**no_checks)
    return H_dict, basis_1d.Ns

def make_Hamiltonian(Hz,N,H_dict):
    params_dict = dict(H0=1.0)
    for i in range(N):
        params_dict["z"+str(i)] = np.random.uniform(-Hz,Hz)
#    print(params_dict)
    H = H_dict.tohamiltonian(params_dict)
    return H

def make_sz(site,N,H_dict):
    params_dict = dict(H0=0.0)
    for i in range(N):
        params_dict["z"+str(i)] = 0.0
    params_dict["z"+str(site)] = 1.0
#    print(params_dict)
    sz = H_dict.tohamiltonian(params_dict)
#    print(sz)
    return sz

def calc_ave_err(vec):
    _n = len(vec)
    _ave = np.sum(vec)/_n
    _var = np.sum((vec-_ave*np.ones(_n))**2)/_n
    _err = np.sqrt(_var/_n)
#    print(_ave,_var,_err)
    return _ave,_err

def main():
    args = parse_args()
    N = args.N
    Nsmp = args.Nsmp
    Hz = args.Hz
#
#    N = 6
    J = 1.0
    twoSz = 0
    rndseed = 12345
#    Nsmp = 10
    print("# N,J,twoSz,rndseed,Nsmp",N,J,twoSz,rndseed,Nsmp)

    H_dict, basis_1d_Ns = make_operator(J,N,twoSz)
#    print("# H_dict",H_dict)
    print("# number_of_states",basis_1d_Ns)
    print("#")
#    print("# Hz mave merr")
    print("# N Nsmp Hz mave merr")

    list_sz = []
    for nsite in range(N):
        for nstate in range(basis_1d_Ns):
            list_sz.append(make_sz(nsite,N,H_dict))
#    print(list_sz)

#    Hzs = np.linspace(0.0,8.0,9)
#    for Hz in Hzs:
#
    m = np.zeros((Nsmp,N,basis_1d_Ns),dtype=np.float)
    dm = np.zeros((Nsmp,N,basis_1d_Ns-1),dtype=np.float)

    for nrnd in range(Nsmp):
        np.random.seed(rndseed+nrnd)
        H = make_Hamiltonian(Hz,N,H_dict)
#        print("# H",H)
        ene, vec = H.eigh()
#        print("# ene",ene)
#        print("# vec",vec)
        for nsite in range(N):
            for nstate in range(basis_1d_Ns):
#                sz = make_sz(nsite,N,H_dict)
                sz = list_sz[nsite*basis_1d_Ns+nstate]
                m[nrnd,nsite,nstate] = np.conjugate(vec[:,nstate]).dot(sz.dot(vec[:,nstate])).real
#                print(nsite,nstate,m[nsite,nstate])
            for nstate in range(basis_1d_Ns-1):
                dm[nrnd,nsite,nstate] = np.abs(m[nrnd,nsite,nstate+1]-m[nrnd,nsite,nstate])
#    print(dm)
#    print(dm.flatten())
#    print(len(dm.flatten()),Nsmp*N*(basis_1d_Ns-1))
    mave, merr = calc_ave_err(dm.flatten())
#    print(Hz,mave,merr)
    print(N,Nsmp,Hz,mave,merr)

if __name__ == "__main__":
    main()
