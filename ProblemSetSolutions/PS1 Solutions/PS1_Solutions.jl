##### Problem Set 1 #####

using JLD2
using Random
using LinearAlgebra
using Statistics
using CSV
using DataFrames
using FreqTables
using Distributions

# NOTE: Remember to set path
cd("/Users/matthew/Documents/Summer University of Oklahoma Course/fall-2020/ProblemSetSolutions")

##### Problem 1
### Set the seed
Random.seed!(1234)

### Part a ###
## i.) Create 10x7 matrix of numbers distr. U~[-5, 10]
A = rand(Uniform(-5, 10), 10, 7)

## ii.) Create 10x7 matrix of numbers distr. N~[-2, 15]
B = rand(Normal(-2, 15), 10, 7)

## iii.) Create 5x7 matrix: first 5r and 5c from A and last two columns of B
C = hcat(A[1:5, 1:5], B[1:5, 6:7])

## iv.) D_i,j = A_i,j if A_i,j <= 0, or 0 otherwise
D = min.(A, 0)

##################################

### Part b ###
## Use built-in Julia function to list the number of elements of A
length(A)
##################################


### Part c ###
## Get number of unique elements of D
length(unique(D))
##################################


### Part d ###
## Use reshape() to create matrix E which is vec operator applied to B
## Can you find an easier way to accomplish this?
E = reshape(B, (length(B), 1))
## Alternative way
vec(B)
##################################


### Part e ###
## Create F, 3-dim that contains A in col 1 of 3rd dim and B in col 2 of 3rd dim
# Can use cat
F = cat(A, B; dims = 3)
##################################


### Part f ###
## Use permutedims() to twist F into a 2x10x7
F = permutedims(F, [3, 2, 1])
##################################


### Part g ###
## Create G, the kronecker product of B and C
## What happens when you try kronecker product of C and F?
G = kron(B, C)
kron(C, F) # This results in an error since F is 3 dimensional
##################################


### Part h ###
## Save matrices A, B, C, D, E, F, and G as .jld file called firstmatrix
# @save from JLD2 package
@save "matrixpractice.jld" A, B, C, D, E, F, G
##################################
