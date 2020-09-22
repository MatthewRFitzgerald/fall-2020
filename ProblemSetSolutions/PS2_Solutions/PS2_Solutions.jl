## Note: Base code is from PS2starter.jl


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 1
#:::::::::::::::::::::::::::::::::::::::::::::::::::
using Optim
f(x) = -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
minusf(x) = x[1]^4+10x[1]^3+2x[1]^2+3x[1]+2
startval = rand(1)   # random starting value
result = optimize(minusf, startval, BFGS())


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 2
#:::::::::::::::::::::::::::::::::::::::::::::::::::
using DataFrames
using CSV
using HTTP
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body)
X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.married.==1

function ols(beta, X, y)
    ssr = (y.-X*beta)'*(y.-X*beta)
    return ssr
end

beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
println(beta_hat_ols.minimizer)

using GLM
bols = inv(X'*X)*X'*y
df.white = df.race.==1
bols_lm = lm(@formula(married ~ age + white + collgrad), df)


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 3
#:::::::::::::::::::::::::::::::::::::::::::::::::::
function logit(alpha, X, d)
    # Create choice probabilities
    #P = exp.(X*alpha)./(1 .+ exp.(X*alpha))
    # Create negative for log likelihood for Optim
    loglike = -(d'*log.(exp.(X*alpha)./(1 .+ exp.(X*alpha))) + (1 .- d)'*log.(1 ./ (1 .+ exp.(X*alpha))))

    #loglike = -( d'*log.(P) + (1 .- d)'*log.(1 .- P) )
    # your turn
    return loglike
end

alpha_hat_logit = optimize(a -> logit(a, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
println(alpha_hat_logit.minimizer)

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 4
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# see Lecture 3 slides for example
alpha_hat_glm = glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 5
#:::::::::::::::::::::::::::::::::::::::::::::::::::
using FreqTables
freqtable(df, :occupation) # note small number of obs in some occupations
df = dropmissing(df, :occupation)
df[df.occupation.==10,:occupation] .= 9
df[df.occupation.==11,:occupation] .= 9
df[df.occupation.==12,:occupation] .= 9
df[df.occupation.==13,:occupation] .= 9
freqtable(df, :occupation) # problem solved

X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.occupation

function mlogit(alpha, X, y)

    # Get number of columns of X
    K = size(X, 2)
    # Get number of discrete values outcome can take
    J = length(unique(y))
    # Get length of outcome vector
    N = length(y) # Get
    # Create array of zeros with N rows and J columns
    bigY = zeros(N,J)
    # Fill bigY with T/F
    for j=1:J
        bigY[:, j] = y.== j
    end
    # Create Kx(J-1) array with an additional column of zeros (i.e. a KxJ array)
    bigAlpha = [reshape(alpha, K, J-1) zeros(K)]

    # Store the promoted type of X and alpha in variable T
    T = promote_type(eltype(X), eltype(alpha))
    # Initialize numerator (array of zeros of type T and size NxJ)
    num = zeros(T, N, J)
    # Initialize denominator (array of zeros of type T and size N)
    dem = zeros(T, N)
    # Create denominator of liklihood's ( 1 + Σ(exp(Xβ)) )
    for j= 1:J
        num[:,j] = exp.(X*bigAlpha[:,j])
        dem .+= num[:,j]
    end

    P = num./repeat(dem, 1, J)

    loglike = -sum( bigY.*log.(P) )

    # your turn

    return loglike
end

alpha_zero = zeros(8*size(X,2))
alpha_rand = rand(8*size(X,2))
alpha_true = [2.6426044235007686, -0.03584576152694649, 0.21827746761254926, -1.558552005365146, 2.327946752730602, -0.03881870330790051, 0.8879450104870784, -2.385181995237595, 3.216454913157838, -0.013261400484619459, 0.0719516111625982, -3.4680139303471886, 0.2541043815380488, -0.008312832852925252, 0.9523388136384844, -2.9362913635370105, 1.1198949001324159, -0.01681635434813123, -0.46869379784793186, -3.493484491780286, 2.7717860852092993, -0.009105361530788308, -1.009981864801167, -5.813613932149742, -5.306050441181659, 0.14394035220035473, -1.3959625883076103,
 -16.71729, 3.047838915333801, -0.020156086187477475, -0.5586905548633996, -4.1336372137203305]
alpha_start = alpha_true.*rand(size(alpha_true))
println(size(alpha_true))
alpha_hat_optim = optimize(a -> mlogit(a, X, y), alpha_start, LBFGS(), Optim.Options(g_tol = 1e-5, iterations = 100_000, show_trace = true, show_every = 50))
alpha_hat_mle = alpha_hat_optim.minimizer
println(alpha_hat_mle)
