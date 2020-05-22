# examples of exact diagonalization calculation by QuSpin

* Use [QuSpin](https://github.com/weinbe58/QuSpin)

* Examples
  * Square FM transverse field Ising model, real time dynamics, sudden quench (focus on the sector "kxblock=(T_x,0),kyblock=(T_y,0),pxblock=(P_x,0),pyblock=(P_y,0),zblock=(Z,0)")
    * H=inf --> H=0.1
      * See Fig.2 in https://doi.org/10.21468/SciPostPhys.4.2.013 (neural networks)
    * H=inf --> H=Hc(=3.04438)
      * See Fig.7(b) in https://arxiv.org/abs/2005.03104 (numerical linked cluster expansion + forward propagation of pure states)
      * See also https://doi.org/10.1103/PhysRevB.99.035115 (iPEPS), https://arxiv.org/abs/1912.08828 (neural networks, comparison with iPEPS)
