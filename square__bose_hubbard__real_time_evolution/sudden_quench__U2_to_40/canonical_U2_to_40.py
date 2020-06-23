## https://weinbe58.github.io/QuSpin/examples/user-basis_example2.html#user-basis-example2-label
## https://weinbe58.github.io/QuSpin/downloads/567d8096559c83a92c52a580c93935c1/user_basis_trivial-boson.py
## http://weinbe58.github.io/QuSpin/generated/quspin.operators.hamiltonian.html
## http://weinbe58.github.io/QuSpin/generated/quspin.basis.boson_basis_general.html#quspin.basis.boson_basis_general

## https://doi.org/10.1103/PhysRevLett.98.180601

from __future__ import print_function, division
from quspin.operators import hamiltonian # Hamiltonians and operators
#from quspin.basis import boson_basis_1d # Hilbert space spin basis_1d
from quspin.basis import boson_basis_general # spin basis constructor
import numpy as np
#import scipy.sparse
import scipy.sparse.linalg
#import scipy as scipy
#import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

###### define model parameters ######
Lx, Ly = 3, 3 # linear dimension of 2d lattice
#Lx, Ly = 3, 4 # linear dimension of 2d lattice
#Lx, Ly = 4, 4 # linear dimension of 2d lattice
N_2d = Lx*Ly # number of sites
N_sps=10 # states per site
Nb=N_2d # total number of bosons
print("Lx",Lx)
print("Ly",Ly)
print("N_2d",N_2d)
print("N_sps",N_sps)
print("Nb",Nb)
###### setting up user-defined symmetry transformations for 2d lattice ######
s = np.arange(N_2d) # sites [0,1,2,....]
x = s%Lx # x positions for sites
y = s//Lx # y positions for sites
T_x = (x+1)%Lx + Lx*y # translation along x-direction
T_y = x +Lx*((y+1)%Ly) # translation along y-direction
P_x = x + Lx*(Ly-y-1) # reflection about x-axis
P_y = (Lx-x-1) + Lx*y # reflection about y-axis
###### setting up bases ######
basis_2d = boson_basis_general(N_2d,Nb=Nb,sps=N_sps,kxblock=(T_x,0),kyblock=(T_y,0),pxblock=(P_x,0),pyblock=(P_y,0))
###### setting up hamiltonian ######
#
J0=0.0 # hopping matrix element
U0=1.0 # onsite interaction
hopping0=[[-J0,j,T_x[j]] for j in range(N_2d)] + [[-J0,j,T_y[j]] for j in range(N_2d)]
interaction0=[[0.5*U0,j,j] for j in range(N_2d)]
potential0=[[-0.5*U0,j] for j in range(N_2d)]
static0=[["+-",hopping0],["-+",hopping0],["nn",interaction0],["n",potential0]]
dynamic0=[]
no_checks=dict(check_symm=False, check_pcon=False, check_herm=False)
H0=hamiltonian(static0,dynamic0,static_fmt="csr",basis=basis_2d,dtype=np.float64,**no_checks).tocsr(time=0)
#H0=hamiltonian(static0,dynamic0,static_fmt="csr",basis=basis_2d,dtype=np.float64).tocsr(time=0)
#print(H0)
#
J1=1.0 # hopping matrix element
U1=0.0 # onsite interaction
hopping1=[[-J1,j,T_x[j]] for j in range(N_2d)] + [[-J1,j,T_y[j]] for j in range(N_2d)]
interaction1=[[0.5*U1,j,j] for j in range(N_2d)]
potential1=[[-0.5*U1,j] for j in range(N_2d)]
static1=[["+-",hopping1],["-+",hopping1],["nn",interaction1],["n",potential1]]
dynamic1=[]
no_checks=dict(check_symm=False, check_pcon=False, check_herm=False)
H1=hamiltonian(static1,dynamic1,static_fmt="csr",basis=basis_2d,dtype=np.float64,**no_checks).tocsr(time=0)
#H1=hamiltonian(static1,dynamic1,static_fmt="csr",basis=basis_2d,dtype=np.float64).tocsr(time=0)
#print(H1)
#
J2=0.0 # hopping matrix element
U2=0.0 # onsite interaction
V2=1.0 # n.n. interaction
hopping2=[[-J2,j,T_x[j]] for j in range(N_2d)] + [[-J2,j,T_y[j]] for j in range(N_2d)]
interaction2=[[0.5*U2,j,j] for j in range(N_2d)]
interactionV2=[[V2,j,T_x[j]] for j in range(N_2d)] + [[V2,j,T_y[j]] for j in range(N_2d)]
interaction2.extend(interactionV2)
potential2=[[-0.5*U2,j] for j in range(N_2d)]
static2=[["+-",hopping2],["-+",hopping2],["nn",interaction2],["n",potential2]]
dynamic2=[]
no_checks=dict(check_symm=False, check_pcon=False, check_herm=False)
H2=hamiltonian(static2,dynamic2,static_fmt="csr",basis=basis_2d,dtype=np.float64,**no_checks).tocsr(time=0)
#H2=hamiltonian(static2,dynamic2,static_fmt="csr",basis=basis_2d,dtype=np.float64).tocsr(time=0)
#print(H2)
#
Ji=1.0 # hopping matrix element
Ui=2.0 # onsite interaction
hoppingi=[[-Ji,j,T_x[j]] for j in range(N_2d)] + [[-Ji,j,T_y[j]] for j in range(N_2d)]
interactioni=[[0.5*Ui,j,j] for j in range(N_2d)]
potentiali=[[-0.5*Ui,j] for j in range(N_2d)]
statici=[["+-",hoppingi],["-+",hoppingi],["nn",interactioni],["n",potentiali]]
dynamici=[]
no_checks=dict(check_symm=False, check_pcon=False, check_herm=False)
Hi=hamiltonian(statici,dynamici,static_fmt="csr",basis=basis_2d,dtype=np.float64,**no_checks).tocsr(time=0)
#Hi=hamiltonian(statici,dynamici,static_fmt="csr",basis=basis_2d,dtype=np.float64).tocsr(time=0)
#print(Hi)
#
Jf=1.0 # hopping matrix element
Uf=40.0 # onsite interaction
hoppingf=[[-Jf,j,T_x[j]] for j in range(N_2d)] + [[-Jf,j,T_y[j]] for j in range(N_2d)]
interactionf=[[0.5*Uf,j,j] for j in range(N_2d)]
potentialf=[[-0.5*Uf,j] for j in range(N_2d)]
staticf=[["+-",hoppingf],["-+",hoppingf],["nn",interactionf],["n",potentialf]]
dynamicf=[]
no_checks=dict(check_symm=False, check_pcon=False, check_herm=False)
Hf=hamiltonian(staticf,dynamicf,static_fmt="csr",basis=basis_2d,dtype=np.float64,**no_checks).tocsr(time=0)
#Hf=hamiltonian(staticf,dynamicf,static_fmt="csr",basis=basis_2d,dtype=np.float64).tocsr(time=0)
#print(Hf)
#
# diagonalise Hi
ene0,vec0 = scipy.sparse.linalg.eigsh(Hi,which='SA',k=2)
print("J U ene/N_2d",J0,U0,ene0[0]/N_2d)
#print(vec0[:,0])


timei = 0.0
timef = 2.0
#timei = 1e-2
#timef = 1e2
dt = 0.01
Nsteps = int(timef/dt+0.1)+1
list_time = [timei+i*(timef-timei)/(Nsteps-1) for i in range(Nsteps)]

#timei = 1e-2
#timef = 1e2
#tratio = 1.01
#Nsteps = int(np.log(timef/timei)/np.log(tratio)+0.1)+1
#list_time = [timei*tratio**i for i in range(Nsteps)]

print("timei",timei)
print("timef",timef)
print("Nsteps",Nsteps)

list_norm2 = []
list_enef = []
list_ene0 = []
list_ene1 = []
list_ene2 = []

ret = vec0[:,0]

#norm2 = np.linalg.norm(ret)**2
#enef = (np.conjugate(ret).dot(Hf.dot(ret)) / norm2).real / N_2d
#ene0 = (np.conjugate(ret).dot(H0.dot(ret)) / norm2).real / N_2d
#ene1 = (np.conjugate(ret).dot(H1.dot(ret)) / norm2).real / N_2d
#ene2 = (np.conjugate(ret).dot(H2.dot(ret)) / norm2).real / N_2d
#list_norm2.append(norm2)
#list_enef.append(enef)
#list_ene0.append(ene0)
#list_ene1.append(ene1)
#list_ene2.append(ene2)

dt = list_time[0]
ret = (scipy.sparse.linalg.expm_multiply((-1j)*dt*Hf,ret,start=0.0,stop=1.0,num=2,endpoint=True))[1]
norm2 = np.linalg.norm(ret)**2
enef = (np.conjugate(ret).dot(Hf.dot(ret)) / norm2).real / N_2d
ene0 = (np.conjugate(ret).dot(H0.dot(ret)) / norm2).real / N_2d
ene1 = (np.conjugate(ret).dot(H1.dot(ret)) / norm2).real / N_2d
ene2 = (np.conjugate(ret).dot(H2.dot(ret)) / norm2).real / N_2d
list_norm2.append(norm2)
list_enef.append(enef)
list_ene0.append(ene0)
list_ene1.append(ene1)
list_ene2.append(ene2)

for i in range(1,Nsteps):
    dt = list_time[i] - list_time[i-1]
    ret = (scipy.sparse.linalg.expm_multiply((-1j)*dt*Hf,ret,start=0.0,stop=1.0,num=2,endpoint=True))[1]
    norm2 = np.linalg.norm(ret)**2
    enef = (np.conjugate(ret).dot(Hf.dot(ret)) / norm2).real / N_2d
    ene0 = (np.conjugate(ret).dot(H0.dot(ret)) / norm2).real / N_2d
    ene1 = (np.conjugate(ret).dot(H1.dot(ret)) / norm2).real / N_2d
    ene2 = (np.conjugate(ret).dot(H2.dot(ret)) / norm2).real / N_2d
    list_norm2.append(norm2)
    list_enef.append(enef)
    list_ene0.append(ene0)
    list_ene1.append(ene1)
    list_ene2.append(ene2)

print("list_time",list_time)
print("list_norm2",list_norm2)
print("list_enef",list_enef)
print("list_ene0",list_ene0)
print("list_ene1",list_ene1)
print("list_ene2",list_ene2)


fig10 = plt.figure()
fig10.suptitle("ene")
plt.plot(list_time,list_enef)
#plt.xscale("log")
plt.xlabel("$t$")
fig10.savefig("fig_N"+str(N_2d)+"_ene.png")

fig20 = plt.figure()
fig20.suptitle("ene_int")
plt.plot(list_time,list_ene0)
#plt.xscale("log")
plt.xlabel("$t$")
fig20.savefig("fig_N"+str(N_2d)+"_ene_int.png")

fig30 = plt.figure()
fig30.suptitle("ene_hop")
plt.plot(list_time,list_ene1)
#plt.xscale("log")
plt.xlabel("$t$")
fig30.savefig("fig_N"+str(N_2d)+"_ene_hop.png")

fig40 = plt.figure()
fig40.suptitle("ene_V")
plt.plot(list_time,list_ene2)
#plt.xscale("log")
plt.xlabel("$t$")
fig40.savefig("fig_N"+str(N_2d)+"_ene_V.png")

fig50 = plt.figure()
fig50.suptitle("norm^2")
plt.plot(list_time,list_norm2)
#plt.xscale("log")
plt.xlabel("$t$")
fig50.savefig("fig_N"+str(N_2d)+"_norm2.png")

fig60 = plt.figure()
fig60.suptitle("<a^{dag}(0)a(1)>")
plt.plot(list_time,(-0.25)*np.array(list_ene1))
#plt.xscale("log")
plt.xlabel("$t$")
fig60.savefig("fig_N"+str(N_2d)+"_adaga.png")

#plt.show()
