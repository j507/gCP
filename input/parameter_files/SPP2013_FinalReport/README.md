# SPP2013 - Final report

## Objectives

- Objective 1
- Objective 2

## Synopsis of the test case

Wonderful synopsis

## Synopsis of the results

- Result 1
- Result 2

## Nomenclature for the .prm files

**Note:** Parameter files have to be run with ./SPP2013

### File name

File names follow the format XX_XX_XX_XX_XX, where XX stands for a variable's value

In this folder the variables are from left to right

1. Mesh identifier (integer)
2. Damage accumulation constant (double)
3. Regularization parameter (double)
4. Degradation exponent (integer)
5. Number of cycles (integer)

**Note:** Not all file names will necessary contain all variables. Due to redundancy or forgetfulness (Most likely the latter)

### Variable's value

- Negative numbers are marked by a leading lower case m, e.g., $-10$ is written as $ \texttt{m10} $. A leading lower case p may be added for readability, e.g., $ 10 $ is then written as $ \texttt{p10} $
- Integers are written as they are. Leading zeros may be added for readability
- Doubles are written in scientific notation, e.g., $ 3\times 10^{-2} $ is written as $ \texttt{3Em2} $. Trailing zeros may be added for readability
- Boolean are explicitly written as $ \texttt{true} $ or $ \texttt{false} $
- Diferent input files, e.g., meshes are to be enumerated and listed in
this file

