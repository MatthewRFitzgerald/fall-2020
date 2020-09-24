


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 1
#:::::::::::::::::::::::::::::::::::::::::::::::::::
using DataFrames
using CSV
using HTTP
using Optim
using LinearAlgebra # For diag function

url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS3-gev/nlsw88w.csv"

df = CSV.read(HTTP.get(url).body)
X = [df.age df.white df.collgrad]
Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
         df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
y = df.occupation


function mlogit_with_Z(theta, X, Z, y)

    # Define alpha as all elements of theta except the coeficient of Z
    alpha = theta[1:end-1]
    # Define gamma as the coefficient on Z
    gamma = theta[end]
    # Define K as the second dimension of X
    K = size(X, 2)
    # Define J as the number of alternatives to choose from
    J = length(unique(y))
    # Define N as the number of observations in the dependent variable
    N = length(y)
    # Define bigY as an array of Zeros of size NxJ
    bigY = zeros(N, J)

    # Fill each column of bigY with indicator = 1 if col_i = choice_i
    for j=1:J
        bigY[:,j] = y.==j
    end
    #=
    Create Kx(J-1) array with an additional column of zeros (i.e. a KxJ array).
    Note this takes alpha (a vector) and transforms it into a KxJ array.
    =#
    bigAlpha = [reshape(alpha, K, J-1) zeros(K)]

    # Store the promoted type of X and alpha in variable T
    T = promote_type(eltype(X), eltype(theta))
    # Initialize numerator (array of zeros of type T and size NxJ)
    num = zeros(T, N, J)
    # Initialize denominator
    dem = zeros(T,N)
    #Create loop
    for j=1:J
        #= Each column i of num refers to the numerator of the choice probability for choice i
        =#
        num[:,j] = exp.(X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*gamma)
        # The denominator is the sum of each of the choice probability numerators
        dem .+= num[:,j]
    end

    #= Create matrix in which every column i is a choice probability for alternative i. Need to divide each column of num by dem which is a vector, so repeat dem J times to allow element-wise division.
    =#
    P = num./repeat(dem,1,J)

    loglike = -sum( bigY.*log.(P) )

    return loglike
end

# Create a 22 element array of starting values
startvals = [2*rand(7*size(X, 2)).-1; .1]
# Declare function is twice differentiable
td = TwiceDifferentiable(theta -> mlogit_with_Z(theta, X, Z, y), startvals; autodiff = :forward)

# Run the optimizer (I'll use Newton's method instead of LBFGS)
theta_hat_optim_ad = optimize(td, startvals, Newton(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace = true, show_every=50))
theta_hat_mle_ad = theta_hat_optim_ad.minimizer

# Evaluate the Hessian at the estimates
H = Optim.hessian!(td, theta_hat_mle_ad)
theta_hat_mle_ad_se = sqrt.(diag(inv(H)))
println([theta_hat_mle_ad theta_hat_mle_ad_se]) # these standard errors match Stata


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 2
#:::::::::::::::::::::::::::::::::::::::::::::::::::

#= The coefficient gamma represents the change in utility with a 1-unit change in log wages. More properly, gamma/100 is the change in utility with a 1% increase in expected wage
=#


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 3
#:::::::::::::::::::::::::::::::::::::::::::::::::::

function nested_logit_with_Z(theta, X, Z, y, nesting_stucture)

    #= Unwrap parameter vector. Alpha contains the coefficients on the x's, lambda contains the measures of the degree of independence in unobserved utility among the alternatives within each nest, and gamma is the coefficient on the difference in Z's
    =#
    alpha = theta[1:end-3]
    lambda = theta[end-2:end-1]
    gamma = theta[end]
    K = size(X,2)
    J = length(unique(y))
    N = length(y)
    bigY = zeros(N,J)
    for j = 1:J
        bigY[:,j] = y.==j
    end
    
