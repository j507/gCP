# Analysis of the Newton's convergence

## Objectives

- Determine the cause of the poor convergence of the Newton-Raphson method

## Synopsis of the test case

Simple shear in 2-D (Plante strain) of an infinite strip. Two symmetric slip systems. Single crystal and polycrystal (3) with no decohesion between grains. In the latter the grain have the same grain orientation or the middle one is misaligned.

## Synopsis of the results

- Poor convergence is due the region of the sigmoid function where the initial linear behavior changes rapidly to its asymptotic one. Here the tangent changes too drastically; thus, the convergence locus is greatly reduced
- A smaller time step reduces the problematic region but seems to also decrease the convergence locus.
- In the single crystal or aligned polycrystal case the slip system activates throughout the domain simultaneously; thus, poor convergence occurs only once. Wheareas in the misaligned polycrystal case one crystal flow uniformly, while the others slowly active due to the dislocation pile-up at the grain boundaries

## Nomenclature for the .prm files

**Note:** Parameter files have to be run with ./simple_shear

### File name

File names follow the format XX_XX_XX, where XX stands for a variable's value

In this folder the variables are from left to right

1. Regularization parameter (double)
2. Load step size (double)
3. Number of equally sized crystals (integer)
4. Whether the crystals have the same orientation or not (boolean)

**Note:** Not all file names will necessary contain all variables. Due to redundancy or forgetfulness (Most likely the latter)

### Variable's value

- Negative numbers are marked by a leading lower case m, e.g., $-10$ is written as $ \texttt{m10} $. A leading lower case p may be added for readability, e.g., $ 10 $ is then written as $ \texttt{p10} $
- Integers are written as they are. Leading zeros may be added for readability
- Doubles are written in scientific notation, e.g., $ 3\times 10^{-2} $ is written as $ \texttt{3Em2} $. Trailing zeros may be added for readability
- Boolean are explicitly written as $ \texttt{true} $ or $ \texttt{false} $
- Diferent input files, e.g., meshes are to be enumerated and listed in
this file

