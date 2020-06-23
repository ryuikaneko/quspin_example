## https://weinbe58.github.io/QuSpin/examples/user-basis_example2.html#user-basis-example2-label
## https://weinbe58.github.io/QuSpin/downloads/567d8096559c83a92c52a580c93935c1/user_basis_trivial-boson.py
## http://weinbe58.github.io/QuSpin/generated/quspin.operators.hamiltonian.html

## https://doi.org/10.1103/PhysRevB.99.054307 --> open BC
## consider periodic BC here

from __future__ import print_function, division
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import boson_basis_1d # Hilbert space spin basis_1d
import numpy as np
#import scipy.sparse
import scipy.sparse.linalg
#import scipy as scipy
#import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

###### define model parameters ######
#N=14 # lattice sites
N=10 # lattice sites
#N_sps=3 # states per site
N_sps=10 # states per site
Nb=N # total number of bosons
print("N",N)
print("N_sps",N_sps)
print("Nb",Nb)
###### setting up bases ######
basis_1d=boson_basis_1d(N,Nb=Nb,sps=N_sps,kblock=0,pblock=1) 
###### setting up hamiltonian ######
#
J0=0.0 # hopping matrix element
U0=3.01 # onsite interaction
hopping0=[[-J0,j,(j+1)%N] for j in range(N)]
interaction0=[[0.5*U0,j,j] for j in range(N)]
potential0=[[-0.5*U0,j] for j in range(N)]
static0=[["+-",hopping0],["-+",hopping0],["nn",interaction0],["n",potential0]]
dynamic0=[]
no_checks=dict(check_symm=False, check_pcon=False, check_herm=False)
H0=hamiltonian(static0,dynamic0,static_fmt="csr",basis=basis_1d,dtype=np.float64,**no_checks).tocsr(time=0)
#H0=hamiltonian(static0,dynamic0,static_fmt="csr",basis=basis_1d,dtype=np.float64).tocsr(time=0)
#print(H0)
#
J1=1.0 # hopping matrix element
U1=0.0 # onsite interaction
hopping1=[[-J1,j,(j+1)%N] for j in range(N)]
interaction1=[[0.5*U1,j,j] for j in range(N)]
potential1=[[-0.5*U1,j] for j in range(N)]
static1=[["+-",hopping1],["-+",hopping1],["nn",interaction1],["n",potential1]]
dynamic1=[]
no_checks=dict(check_symm=False, check_pcon=False, check_herm=False)
H1=hamiltonian(static1,dynamic1,static_fmt="csr",basis=basis_1d,dtype=np.float64,**no_checks).tocsr(time=0)
#H1=hamiltonian(static1,dynamic1,static_fmt="csr",basis=basis_1d,dtype=np.float64).tocsr(time=0)
#print(H1)
#
J2=0.0 # hopping matrix element
U2=0.0 # onsite interaction
V2=1.0 # n.n. interaction
hopping2=[[-J2,j,(j+1)%N] for j in range(N)]
interaction2=[[0.5*U2,j,j] for j in range(N)]
interactionV2=[[V2,j,(j+1)%N] for j in range(N)]
interaction2.extend(interactionV2)
potential2=[[-0.5*U2,j] for j in range(N)]
static2=[["+-",hopping2],["-+",hopping2],["nn",interaction2],["n",potential2]]
dynamic2=[]
no_checks=dict(check_symm=False, check_pcon=False, check_herm=False)
H2=hamiltonian(static2,dynamic2,static_fmt="csr",basis=basis_1d,dtype=np.float64,**no_checks).tocsr(time=0)
#H2=hamiltonian(static2,dynamic2,static_fmt="csr",basis=basis_1d,dtype=np.float64).tocsr(time=0)
#print(H2)
#
J=1.0 # hopping matrix element
U=3.01 # onsite interaction
hopping=[[-J,j,(j+1)%N] for j in range(N)]
interaction=[[0.5*U,j,j] for j in range(N)]
potential=[[-0.5*U,j] for j in range(N)]
static=[["+-",hopping],["-+",hopping],["nn",interaction],["n",potential]]
dynamic=[]
no_checks=dict(check_symm=False, check_pcon=False, check_herm=False)
H=hamiltonian(static,dynamic,static_fmt="csr",basis=basis_1d,dtype=np.float64,**no_checks).tocsr(time=0)
#H=hamiltonian(static,dynamic,static_fmt="csr",basis=basis_1d,dtype=np.float64).tocsr(time=0)
#print(H)
#
# diagonalise H1
ene0,vec0 = scipy.sparse.linalg.eigsh(H1,which='SA',k=2)
print("J U ene/N",J1,U1,ene0[0]/N)
#print(vec0[:,0])


##timei = 0.0
##timef = 1.0
#timei = 1e-2
#timef = 1e2
#dt = 0.01
#Nsteps = int(timef/dt+0.1)+1
#list_time = [timei+i*(timef-timei)/(Nsteps-1) for i in range(Nsteps)]

timei = 1e-2
timef = 1e2
tratio = 1.01
Nsteps = int(np.log(timef/timei)/np.log(tratio)+0.1)+1
list_time = [timei*tratio**i for i in range(Nsteps)]

print("timei",timei)
print("timef",timef)
print("Nsteps",Nsteps)

list_norm2 = []
list_ene = []
list_ene0 = []
list_ene1 = []
list_ene2 = []

ret = vec0[:,0]

#norm2 = np.linalg.norm(ret)**2
#ene = (np.conjugate(ret).dot(H.dot(ret)) / norm2).real / N
#ene0 = (np.conjugate(ret).dot(H0.dot(ret)) / norm2).real / N
#ene1 = (np.conjugate(ret).dot(H1.dot(ret)) / norm2).real / N
#ene2 = (np.conjugate(ret).dot(H2.dot(ret)) / norm2).real / N
#list_norm2.append(norm2)
#list_ene.append(ene)
#list_ene0.append(ene0)
#list_ene1.append(ene1)
#list_ene2.append(ene2)

dt = list_time[0]
ret = (scipy.sparse.linalg.expm_multiply((-1j)*dt*H,ret,start=0.0,stop=1.0,num=2,endpoint=True))[1]
norm2 = np.linalg.norm(ret)**2
ene = (np.conjugate(ret).dot(H.dot(ret)) / norm2).real / N
ene0 = (np.conjugate(ret).dot(H0.dot(ret)) / norm2).real / N
ene1 = (np.conjugate(ret).dot(H1.dot(ret)) / norm2).real / N
ene2 = (np.conjugate(ret).dot(H2.dot(ret)) / norm2).real / N
list_norm2.append(norm2)
list_ene.append(ene)
list_ene0.append(ene0)
list_ene1.append(ene1)
list_ene2.append(ene2)

for i in range(1,Nsteps):
    dt = list_time[i] - list_time[i-1]
    ret = (scipy.sparse.linalg.expm_multiply((-1j)*dt*H,ret,start=0.0,stop=1.0,num=2,endpoint=True))[1]
    norm2 = np.linalg.norm(ret)**2
    ene = (np.conjugate(ret).dot(H.dot(ret)) / norm2).real / N
    ene0 = (np.conjugate(ret).dot(H0.dot(ret)) / norm2).real / N
    ene1 = (np.conjugate(ret).dot(H1.dot(ret)) / norm2).real / N
    ene2 = (np.conjugate(ret).dot(H2.dot(ret)) / norm2).real / N
    list_norm2.append(norm2)
    list_ene.append(ene)
    list_ene0.append(ene0)
    list_ene1.append(ene1)
    list_ene2.append(ene2)

print("list_time",list_time)
print("list_norm2",list_norm2)
print("list_ene",list_ene)
print("list_ene0",list_ene0)
print("list_ene1",list_ene1)
print("list_ene2",list_ene2)


fig10 = plt.figure()
fig10.suptitle("ene") 
plt.plot(list_time,list_ene)
plt.xscale("log")
plt.xlabel("$t$")
fig10.savefig("fig_N"+str(N)+"_ene.png")

fig20 = plt.figure()
fig20.suptitle("ene_int") 
plt.plot(list_time,list_ene0)
plt.xscale("log")
plt.xlabel("$t$")
fig20.savefig("fig_N"+str(N)+"_ene_int.png")

fig30 = plt.figure()
fig30.suptitle("ene_hop") 
plt.plot(list_time,list_ene1)
plt.xscale("log")
plt.xlabel("$t$")
fig30.savefig("fig_N"+str(N)+"_ene_hop.png")

fig40 = plt.figure()
fig40.suptitle("ene_V") 
plt.plot(list_time,list_ene2)
plt.xscale("log")
plt.xlabel("$t$")
fig40.savefig("fig_N"+str(N)+"_ene_V.png")

fig50 = plt.figure()
fig50.suptitle("norm^2") 
plt.plot(list_time,list_norm2)
plt.xscale("log")
plt.xlabel("$t$")
fig50.savefig("fig_N"+str(N)+"_norm2.png")

#plt.show()
