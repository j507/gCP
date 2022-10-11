# Gradient crystal plasticity solver

[![CI](https://github.com/j507/gCP/actions/workflows/main.yml/badge.svg)](https://github.com/j507/gCP/actions/workflows/main.yml)

Solver for crystal plasticity problems based on the gradient-enhanced formulation proposed by [M. Gurtin](https://doi.org/10.1016/S0022-5096(99)00059-9). Only small deformations are currently supported.

Known problems:
- Low values of the energetic length scale lead to divergence near the rate-independent case (See issue #1)
- Problems considering polycrystals can only be run in serial (See issue #2)
- Method of marking cells at grain boundaries does not work properly in parallel (See issue #3)
- The cohesive law introduces a normal vector mismatch between cells at the interface. 
- Cell penetration is not prevented in the current code. 

Yet to be implemented:
- Dimensionless formulation
- Penalty term 
- Cohesive element
- Node-to-segment contact discretization
- Microhard boundary conditions
- Neumann boundary conditions at the domain's boundary
