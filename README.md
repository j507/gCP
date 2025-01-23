# Strain-gradient crystal (visco-)plasticity solver

[![CI](https://github.com/j507/gCP/actions/workflows/main.yml/badge.svg)](https://github.com/j507/gCP/actions/workflows/main.yml)

Solver for crystal plasticity problems based on the gradient-enhanced formulation proposed by [M. Gurtin](https://doi.org/10.1016/S0022-5096(99)00059-9) and enhanced by an cohesive zone model at the grain boundaries. Only small deformations are currently supported.

Known problems:
- Problems considering polycrystals can only be run in serial (See issue #2)
- Most parameter files have yet to be updated to the master branch. Default .prm files serve as reference