# Gradient crystal plasticity solver

[![CI](https://github.com/j507/gCP/actions/workflows/main.yml/badge.svg)](https://github.com/j507/gCP/actions/workflows/main.yml)

Solver for crystal plasticity problems based on the gradient-enhanced formulation proposed by [M.Gurtin](https://doi.org/10.1016/S0022-5096(99)00059-9). Only small deformations are currently supported.

Known problems:
- Low values of the energetic length scale lead to divergence near the rate-independent case (See issue #1)
- Problems considering polycrystals can only be run in serial (See issue #2)

Yet to be implemented:
- Dimensionless formulation
- Microhard boundary conditions
- Neumann boundary conditions at the domain's boundary
- Decohesion model
