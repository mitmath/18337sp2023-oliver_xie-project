import Pkg; Pkg.add("DifferentialEquations");
Pkg.add("DiffEqOperators");
Pkg.add("LinearAlgebra");
Pkg.add("ApproxFun");
Pkg.add("Sundials");
Pkg.add("Plots");
Pkg.add("Zygote")
Pkg.add("Polynomials");
Pkg.add("CSV");
Pkg.add("DataFrames")

using DifferentialEquations, DiffEqOperators, LinearAlgebra, ApproxFun, Sundials, Plots, Zygote, Polynomials, CSV, DataFrames, Printf

cd("C:/Users/ChemeGrad2020/Dropbox (MIT)/Coursework/2023_Spring/18337")
ft = CSV.read("m_48_1_1_ft.csv", DataFrame, header=false)
fs = CSV.read("m_48_1_1_fs.csv", DataFrame, header=false)
gsq = CSV.read("m_48_1_1_Gsq.csv", DataFrame, header=false)
psi_hat = CSV.read("f45_lam_psiS.csv", DataFrame, header=false)

FT = Matrix(ft)
FS = Matrix(fs)
Gsq = Matrix(gsq)
P̂ = Matrix(psi_hat)

ntau = 25
np = 100 # number of Chebyshev coefficients (parameters) in each spatial point
#x = points(S, n)

# Convert the initial condition to Fourier space
q₀ = FT * ones(ntau)
L = vec(Gsq)

Diff = 1 / 6
Afwd = DiffEqArrayOperator(Diagonal(-Diff .* L))
Arev = DiffEqArrayOperator(Diagonal(Diff .* L))

function flat(p_full)
    p = vec(p_full)
    return p
end

function unflatten(p)
    # unflattens the p into n columns of Chebyshev coefficients
    p_full = reshape(p, ntau, np)
    return p_full
end

function potential(p, s)
    # p is a flat vector
    p_full = unflatten(p)
    w = zeros(ntau)
    for i = 1:ntau
        a = ChebyshevT(p_full[i,:])
        w[i] = a(s)
    end
    return w
end

function fokker_planck_fwd!(du, u, p, s)
    ω, tmp = p(s)
    # Transform u back to point-space
    mul!(tmp, FS, u)
    # apply nonlinear function ω * u
    @. tmp = -ω * tmp
    mul!(du, FT, tmp) # Transform back to Fourier space
    #du = FT * (-ω .* (FS * u))
end

function fokker_planck_rev!(dû, û, p, s)
    ω, tmp = p(s)
    # Transform u back to point-space
    mul!(tmp, FS, û)
    # apply nonlinear function ω * u
    @. tmp = ω * tmp
    mul!(dû, FT, tmp) # Transform back to Fourier space
end

function pullback_fwd(y, u, ω)
    f(u, ω) = Diagonal(-Diff .* L) * u -  FT * (ω .* (FS * u))
    _, back = Zygote.pullback(f, u, ω)
    return back(y)
end

function pullback_rev(y, u, ω)
    f(u, ω) = Diagonal(Diff .* L) * u + FT * (ω .* (FS * u))
    _, back = Zygote.pullback(f, u, ω)
    return back(y)
end

function adjointode2!(dx, x, p, s)
    # dx is composed of first lambda, then q_fwd, then q_rev, then p
    # There are 2*M values of lambda, M values of q_fwd, M values of q_rev, M×Np values of p
    q_fwd, q_rev, a = p(s)
    λ = x[1:2*ntau]
    μ = x[(2*ntau+1):end]
    # Need to first find the Chebyshev Polynomial Matrix for this specific s
    cheby_fwd = zeros(np)
    cheby_rev = zeros(np)
    for i = 1:np
        point_vec = zeros(np)
        point_vec[i] = 1;
        cheby_fwd[i] = ChebyshevT(point_vec)(s)
        cheby_rev[i] = ChebyshevT(point_vec)(1-s)
    end
    # p is matrix of coefficients (M x Np)
    f(q1, q2, p1) = [ Diagonal(-Diff .* L) * q1 - (FS * (Diagonal(p1 * cheby_fwd) * (FT * q1)));
    Diagonal(-Diff .* L) * q2 - (FS * (Diagonal(p1 * cheby_rev) * (FT * q2)))]

    _, back = Zygote.pullback(f, q_fwd, q_rev, a)
    q̄_fwd, q̄_rev, ā = back(λ)
    vjp = [q̄_fwd; q̄_rev; flat(ā)]

    # stupid derivative Calculation
    # matches pullback
    #dq_fwd = zeros(ntau)
    #dq_rev = zeros(ntau)
    #da = zeros(ntau, np)
    #for i = 1:ntau
    #    point_vec = zeros(ntau)
    #    point_vec[i] = 1;
    #    dq_fwd[i] = sum(Diagonal(-Diff .* L) * point_vec - (FS * (Diagonal(cheby_full * cheby_fwd) * (FT * point_vec))))
    #    dq_rev[i] = sum(Diagonal(-Diff .* L) * point_vec - (FS * (Diagonal(cheby_full * cheby_rev) * (FT * point_vec))))
    #end

    dx .= -vjp
end

function loss(ψ_s, ψ₀, Q)
    # Compares loss in T space
    L = sum( (ψ_s - ψ₀).^2 )
    return L
end

function testode!(du, u, p, s)
    # This works
    cheby_fwd = zeros(np)
    cheby_rev = zeros(np)
    for i = 1:np
        point_vec = zeros(np)
        point_vec[i] = 1;
        cheby_fwd[i] = ChebyshevT(point_vec)(s)
        cheby_rev[i] = ChebyshevT(point_vec)(1-s)
    end
    #du .= Diagonal(-Diff .* L) * u - (FS * (Diagonal(p * cheby_fwd) * (FT * u)))
    du .= Diagonal(-Diff .* L) * u - (FS * (Diagonal(p * cheby_rev) * (FT * u)))
end

#savetimes = (0.0:0.001:1.0)
#test_prob = ODEProblem(testode!, q₀, (0.0, 1.0), cheby_full)
#q_test = solve(test_prob, Tsit5(), tstops = savetimes)

# Solve the forward pass and send results to reverse pass
# Define u' = Au + f(ω, u)

# initial guess of cheby_p
#cheby_p = randn(250,1)
cheby_v = randn(np)
cheby_v[1] += 5
#cheby_full = repeat(cheby_v', ntau, 1)
cheby_full = zeros(ntau, np)
cheby_full[:,1] = 10 .* ones(ntau)
cheby_p = flat(cheby_full)

γ = 0.01;
tmax = 100
c = zeros(tmax)

for i = 1:tmax
    savetimes = (0.0:0.01:1.0) # what is the discretization here that gives the most accurate solution
    p(s) = (potential(cheby_p, s), similar(q₀))
    prob_fwd = SplitODEProblem(Afwd, fokker_planck_fwd!, q₀, (0.0, 1.0), p)
    prob_rev = SplitODEProblem(Arev, fokker_planck_rev!, q₀, (1.0, 0.0), p)

    q_fwd = solve(prob_fwd, KenCarp4(), tstops = savetimes)
    q_rev = solve(prob_rev, KenCarp4(), tstops = savetimes)

    Q = q_fwd(1.0)[1]
    psi(s) = 1/Q * (FS*q_fwd(s)) .* (FS*q_rev(s))
    psi_s(s) = FT * psi(s)

    # Flatten results into vector, set up the second parameter struct
    #p_adjoint = s -> (q_fwd = q_fwd(s), q_rev = q_rev(s), ω = potential(cheby_p, s))
    #p_adjoint = s -> (q_fwd = q_fwd(s), ω = potential(cheby_p, s))
    p_adjoint = s -> (q_fwd = q_fwd(s), q_rev = q_rev(1-s), a = unflatten(cheby_p))

    #xf = [vec(q_fwd(1)); vec(q_rev(0)); zeros(ntau)]
    #Cq(q_fwd, q_rev, s) = [2 * (psi_s(s) - P̂[:,convert(Int64, s*100)+1]) * 1/Q .* q_rev(1-s); 2 * (psi_s(s) - P̂[:,convert(Int64, s*100)+1]) * 1/Q .* q_fwd(s)]
    Cq(q_fwd, q_rev, s) = [2 * (psi_s(s) - P̂[:,convert(Int64, round(s*100))+1]) * 1/Q; 2 * (psi_s(s) - P̂[:,convert(Int64, round(s*100))+1]) * 1/Q]
    
    xf = [Cq(q_fwd, q_rev, 1); zeros(ntau*np)]

    checktimes = 0.0:0.05:0.95
    condition(u, t, integrator) = t ∈ checktimes
    affect!(integrator) = integrator.u[1:2*ntau] += Cq(q_fwd, q_rev, integrator.t)
    cb = DiscreteCallback(condition, affect!)

    optimprob = ODEProblem(adjointode2!, xf, (1.0, 0.0), p_adjoint)
    dx = solve(optimprob, Tsit5(), callback = cb, tstops = checktimes)

    # get current cost
    for s_check ∈ checktimes
        c[i] = c[i] + loss(psi_s(s_check), P̂[:, convert(Int64, round(s_check*100))+1], Q)
    end

    # Update
    dcheby_p = dx(0)[2*ntau+1:end]
    cheby_p = cheby_p - γ .* dx(0)[2*ntau+1:end]
    # Levenberg-Marquardt method
    #λ = 100
    #δ = (dcheby_p' * dcheby_p + λ) \ (-c[i])
    #cheby_p = cheby_p + dcheby_p .* δ

    @printf("Iteration %0.0f of %0.0f \n", i, tmax)
end

plot(1:1:tmax, c, label = "Loss function")
