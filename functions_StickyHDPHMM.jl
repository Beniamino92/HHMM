ENV["MPLBACKEND"]="qt4agg"; using PyPlot
using Distributions
using ProgressMeter
using StatsBase
using Optim
using JLD
using DSP
using RCall
using CSV
using DataFrames



# Function: return totSeq, an indSeq, given state sequence z.
function get_Seq(z)

    totSeq = zeros(Int64, Kz)
    indSeq = zeros(Int64, T, Kz)

    for t in 1:T
        # -- Add z_t to count vector and observation statistics:
        totSeq[z[t]] = totSeq[z[t]] + 1
        indSeq[totSeq[z[t]],z[t]] = copy(t)
    end

    out = Dict("totSeq" => totSeq,
               "indSeq" => indSeq)
end

# Function: findpeaks of a 1D array, sorted in decreasing order (of power)
function findpeaks(A, n_peaks)

  maximums = Tuple{Float64, Int}[]
  n = length(A)

  if (A[1] > A[2])
    push!(maximums, (A[1], 1))
  end

  if (A[n] > A[n-1])
    push!(maximums, (A[n], n))
  end

  for j in 2:(n-1)
    if (A[j-1] < A[j] && A[j+1] < A[j])
      push!(maximums, (A[j], j))
    end
  end

  maximums = sort(maximums, by = x->x[1], rev = true)[1:min(n_peaks, length(maximums))]

  return maximums
end

# Function: get maximum value of b for evaluating Binomial(a, i),
#           for i = 1, ...b.
function get_max_b(a, b)
  b_max = 1
  try
    for i in 1:b
      binomial(a, i)
      b_max += 1
    end
    return (b_max - 1)
  catch
    return (b_max - 1)
  end
end

# Function: get p_value for Fisher test, for the largest periodogram ordinate
# if it is not possible to evaluate binom(a, b) it will return (-1).
function get_p_value_freq(x, a::Int64, b::Int64)
  out = 0.0
  for i in 1:b
    out += ((-1)^(i-1))*binomial(a, i)*((1-i*x)^(a-1))
  end
  return out
end

# Function: get_significant ω, by testing w.r.t to true distribution (Fisher test)
# using smoothed periodogram.
function find_significant_ω(y, n_freq_max, sigma_smoothing)

  ω = []
  n = length(y)
  y_demean = y - mean(y)

  n = length(y_demean)
  period_smooth = periodogram(y_demean, window = gaussian(n, sigma_smoothing))

  I_smooth = period_smooth.power[2:(end-1)]
  freq = period_smooth.freq[2:(end-1)]

  freq_test = freq[[x[2] for x in findpeaks(I_smooth, n_freq_max)]]
  I_test = [x[1] for x in findpeaks(I_smooth, n_freq_max)]
  a = floor(Int64, (n-1)/2)

  for i in 1:length(I_test)

    g_test = I_test[i]/sum(I_smooth)
    b = floor(Int64, 1/g_test)
    b_max = get_max_b(a, b)
    x = copy(g_test)
    p_val = get_p_value_freq(x, a, b_max)

    if (p_val <= 5e-2 && p_val >= 0)
      push!(ω, freq_test[i])
    end

  end

  return ω
end

# Function: sample concentration parameters that define the distribution on transition
#           distributions
function sample_hyperparams(N, M, barM, sum_w, α_plus_κ, γ, hyperHMMhyperparams)

    a_α = hyperHMMhyperparams["a_α"]
    b_α = hyperHMMhyperparams["b_α"]
    a_γ = hyperHMMhyperparams["a_γ"]
    b_γ = hyperHMMhyperparams["b_γ"]
    c = hyperHMMhyperparams["c"]
    d = hyperHMMhyperparams["d"]


    Nkdot = sum(N, 2)
    Mkdot = sum(M, 2)
    barK = length(find(sum(barM, 1).>0))
    validindices = find(Nkdot .> 0)

    # Resample concentration parameters -----
    if isempty(validindices)
        α_plus_κ = rand(Gamma(a_α, 1/b_α)) # Gj concentration parameter
        γ = rand(Gamma(a_γ, 1/b_γ)) # G_0 concentration parameter
    else
        α_plus_κ = gibbs_conparam(α_plus_κ, Nkdot[validindices], Mkdot[validindices],
                                 a_α, b_α, 50)
        γ = gibbs_conparam(γ, sum(barM), barK,
                           a_γ, b_γ, 50)
    end

    # Resample self-transition proportion parameter
    ρ = rand(Beta(c + sum(sum_w), d+(sum(sum(M))-sum(sum_w))))

    output = Dict("γ" => γ, "α_plus_κ" => α_plus_κ, "ρ" => ρ)

    return output
end


# Function: sample hyperparameters, starting values.
function sample_hyperparams_init(hyperHMMhyperparams)

    # Hyperparams for Gamma dist over α + κ, where transition distributions
    # π_j ∼ DP (α+κ, (α⋅β + κ δ(j)/(α + κ))), which is the same as
    # π_j ∼ DP (α+κ, (1-ρ)⋅β + ρ⋅δ(j))

    a_α = hyperHMMhyperparams["a_α"]
    b_α = hyperHMMhyperparams["b_α"]

    # Hyperparams for gamma dist over γ, where avg transition distribution
    # β ∼ stick(γ)
    a_γ = hyperHMMhyperparams["a_γ"]
    b_γ = hyperHMMhyperparams["b_γ"]

    # Hyperparams for Beta dist over ρ, where ρ relates α+κ
    # to α and κ individually.
    c = hyperHMMhyperparams["c"]
    d = hyperHMMhyperparams["d"]


    # Resample concentration parameters, from the prior.
    α_plus_κ = a_α / b_α #  Gj concentration parameter
    γ = a_γ / b_γ # G_0 concentration parameter
    ρ = c/(c+d)[1]


    HMMhyperparams = Dict("γ" => γ, "α_plus_κ" => α_plus_κ, "ρ" => ρ)
end


# Function: returns starting values for m, ω, β and σ, given z_start.
function get_starting_value_θ(z_start, n_freq_max_start)

  Seq_start = get_Seq(z_start)
  totSeq_start = Seq_start["totSeq"]
  indSeq_start = Seq_start["indSeq"]

  for j in 1:Kz
    if (totSeq_start[j] > n_min_obs)

      temp_ind = indSeq_start[1:totSeq_start[j], j]
      temp_ind_seg = get_time_indexes(temp_ind)
      info_longest_ts = get_segment_ts(data, temp_ind_seg, false)

      global a = info_longest_ts["time"][1]
      global b = info_longest_ts["time"][end]

      global time_star = collect(a:b)
      global y = info_longest_ts["data"]
      global y_star = copy(y)
      global n = length(y)

      global ω = find_significant_ω(y, n_freq_max_start, σ_smoothing)
      m_sample[j, 1] = length(ω)
      ω_sample[j, 1:length(ω), 1] = copy(ω)
      global σ = 2
      β = optimize(neg_f_posterior_β_stationary, neg_g_posterior_β_stationary!, neg_h_posterior_β_stationary!,
              zeros(2*m_sample[j,1]), NelderMead()).minimizer
      β_sample[j, 1:(2*m_sample[j,1]), 1] = copy(β)
      σ_sample[j, 1] = copy(σ)
    else
      m_sample[j, 1] = 1
      β_sample[j, 1:(2*m_sample[j, 1]), 1] = rand(MultivariateNormal(zeros((2*m_sample[j,1])),
                                                  eye((2*m_sample[j, 1]))))
      ω_sample[j, 1:m_sample[j, 1], 1] = rand(Uniform(0, ϕ_ω))
      σ_sample[j, 1] = rand(InverseGamma(1, 2))
    end
  end
end


# Function: sample transition distributions, given sampled hyperparameters and
#           count matrices
function sample_dist(α_plus_κ, γ, ρ, N, barM)

    # - Define α and κ in terms of (α+κ) and ρ
    α = α_plus_κ*(1-ρ)
    κ = α_plus_κ*ρ

    # Sample β, the global menu, given new barM:
    β_vec = rand(Dirichlet(reshape(sum(barM, 1) + γ/Kz, Kz)))
    if any(β_vec .== 0.0)
      loc_zeros = find(β_vec .== 0)
      β_vec[loc_zeros] = 1e-100
    end

    π_z = zeros(Kz,Kz)

    for j in 1:Kz
        κ_vec = zeros(Kz)
        # Add an amount κ to a Dirichlet parameter corresponding to a
        # self transition.
        κ_vec[j] = copy(κ)
        # Sample π_j's given sampled β_vec and counts N, where
        # DP(α + κ, (α⋅β)/(α+κ)) is  Dirichelet distributed over the finite partition defined by β_vec
        π_z[j, :] = rand(Dirichlet(α*β_vec + κ_vec + N[j, :]))
    end

    π_init = rand(Dirichlet(α*β_vec + N[Kz+1, :]));

    output = Dict("π_z"=>π_z, "π_init"=>π_init, "β_vec"=>β_vec)

    return output
end



# Function: Compute f(y_i ; μ_{y_i}, σ_{y_i}) for i = 1, …, T.
#           It returns a (Kz × T) matrix
function compute_likelihood(m, β, ω, σ)

    log_likelihood = zeros(Float64, Kz, T)
    μ = zeros(Float64, Kz, T)

    for j in 1:Kz
        X = get_X(ω[j, 1:m[j]], 1, T)
        μ[j, :] = X*β[j, 1:(2*m[j])]
    end

    for kz in 1:Kz
        u = (1/σ[kz])*(data - μ[kz, :])
        log_likelihood[kz, :] = -0.5*(u.^2) - log(σ[kz])
        log_likelihood[kz, find(isnan.(log_likelihood[kz, :]))] = -1e100
    end

    normalizer = [maximum(log_likelihood[:, i]) for i =1:T]
    log_likelihood = log_likelihood - reshape(repeat(normalizer, inner = [Kz, 1]), Kz, T)
    likelihood = exp.(log_likelihood)

    return likelihood
end



# Function: compute backward messages.
function backwards_message_vec(likelihood, π_z)

    bwds_msg = ones(Float64, Kz, T)
    partial_marg = zeros(Kz, T)

    # Compute messages bacwards in time
    for t in (T-1):-1:1
        # Mutiplying likelihood by incoming message
        partial_marg[:, t+1] = likelihood[:, t+1].*bwds_msg[:, t+1]

        # Integrate out z_t
        bwds_msg[:, t] = π_z * partial_marg[:, t+1]
        bwds_msg[:, t] = bwds_msg[:, t]/sum(bwds_msg[:, t]);
    end

    # Compute marginal for first time point
    partial_marg[:, 1] = likelihood[:, 1] .* bwds_msg[:, 1]

    return  partial_marg
end


# Function: sample the mode z's, given the observations, transition distributions
#           and emission parameters.
function sample_z(π_z, π_init, m, β, ω, σ)

    N = zeros(Int64, Kz + 1, Kz)
    z = zeros(Int64, T)
    totSeq = zeros(Int64, Kz)
    indSeq = zeros(Int64, T, Kz)

    # Compute likelihood of each observation under each parameter θ
    likelihood = compute_likelihood(m, β, ω, σ)
    # Compute backward messages
    partial_marg = backwards_message_vec(likelihood, π_z)


    for t in 1:T

        # ---  Sample z(t)
        if (t == 1)
            Pz = π_init .* partial_marg[:, 1]
        else
            Pz = π_z[z[t-1], :] .* partial_marg[:, t];
        end
        Pz = cumsum(Pz)
        z[t] = 1 + sum(Pz[end]*rand() .> Pz)

        # --- Add state to counts matrix
        if (t > 1)
            N[z[t-1], z[t]] = N[z[t-1], z[t]] + 1
        else
            N[Kz+1, z[t]] = N[Kz+1, z[t]] + 1
        end

        # Add z_t to count vector and observation statistics:
        totSeq[z[t]] = totSeq[z[t]] + 1
        indSeq[totSeq[z[t]],z[t]] = copy(t)
    end


    output = Dict("stateSeq" => z, "N" => N,
                  "totSeq" => totSeq, "indSeq" => indSeq)

    return output
end


# Function: sample number of tables that served dish k in restaurant j
#           for each k and j. It returns a (Kz+1)×(Kz) matrix
function randnumtable(α, numdata)

    numtable = zeros(Int64, size(numdata))

    for ii = 1:prod(size(numdata))
        if (numdata[ii] > 0)
            numtable[ii] = 1 + sum(rand(numdata[ii] - 1) .<
                ones(Float64, numdata[ii] - 1)*α[ii]./(α[ii]+(1:numdata[ii]-1)))
        else
            numtable[ii] = 1
        end
    end
    numtable[numdata .== 0] = 0

    return numtable
end



# - Function: sample barM and sum_w
function sample_barM(M, β_vec, ρ)

    barM = copy(M)
    sum_w = zeros(Int64, Kz)

    for j in 1:Kz
        p = ρ/(β_vec[j]*(1-ρ) + ρ)
        sum_w[j] = rand(Binomial(M[j,j], p))
        barM[j,j] = M[j,j] - sum_w[j]
    end

    output = Dict("sum_w" => sum_w,
                  "barM" => barM)
    return output
end


# Function: sample_tables
function sample_tables(N, α_plus_κ, ρ, β_vec)

    # - Define α and κ in terms of (α+κ) and ρ
    α = α_plus_κ*(1-ρ)
    κ = α_plus_κ*ρ

    # Sample M, where M(i,j) = # of tables in restaurant i served dish j:
    M = randnumtable([α*repeat(β_vec', outer = [Kz, 1]) +  κ*eye(Kz); α*β_vec'], N)

    # Sample barM (the table counts for the underlying restaurant), where
    # barM(i,j) = # tables in restaurant i that considered dish j:
    temp = sample_barM(M, β_vec, ρ)
    barM = temp["barM"]
    sum_w = temp["sum_w"]

    output = Dict("M" => M, "barM" => barM, "sum_w" => sum_w)

    return output
end



# Function: Auxiliary variable resampling of DP concentration parameter
function gibbs_conparam(α, numdata, numclass, aa, bb, numiter)

    numgroup = length(numdata)
    totalclass = sum(numclass)

    for ii in 1:numiter

        #  beta auxiliary variables
        xx = [rand(Beta(α + 1, numdata[j])) for j=1:length(numdata)]

        # binomial auxiliary variables
        zz = convert(Array{Int64}, (rand(numgroup) .* (α + numdata)) .< numdata)

        # gamma resampling of concentration parameter
        gammaa = aa + totalclass - sum(zz)
        gammab = bb - sum(log.(xx))
        α = rand(Gamma(gammaa, 1/gammab))

    end

    return α
end



# Function: sample concentration parameters that define the distribution on transition
#           distributions
function sample_hyperparams(N, M, barM, sum_w, α_plus_κ, γ, hyperHMMhyperparams)

    a_α = hyperHMMhyperparams["a_α"]
    b_α = hyperHMMhyperparams["b_α"]
    a_γ = hyperHMMhyperparams["a_γ"]
    b_γ = hyperHMMhyperparams["b_γ"]
    c = hyperHMMhyperparams["c"]
    d = hyperHMMhyperparams["d"]


    Nkdot = sum(N, 2)
    Mkdot = sum(M, 2)
    barK = length(find(sum(barM, 1).>0))
    validindices = find(Nkdot .> 0)

    # Resample concentration parameters -----
    if isempty(validindices)
        α_plus_κ = rand(Gamma(a_α, 1/b_α)) # Gj concentration parameter
        γ = rand(Gamma(a_γ, 1/b_γ)) # G_0 concentration parameter
    else
        α_plus_κ = gibbs_conparam(α_plus_κ, Nkdot[validindices], Mkdot[validindices],
                                 a_α, b_α, 50)
        γ = gibbs_conparam(γ, sum(barM), barK,
                           a_γ, b_γ, 50)
    end

    # Resample self-transition proportion parameter
    ρ = rand(Beta(c + sum(sum_w), d+(sum(sum(M))-sum(sum_w))))

    output = Dict("γ" => γ, "α_plus_κ" => α_plus_κ, "ρ" => ρ)

    return output
end



# -- Function: given a sequence of indexes it returns a matrix with
#    start point and end point of each segment of time series corresponding
#    to that regime.

function get_time_indexes(indexes)

  A = zeros(Int64, 2)
  for t in 1:(length(indexes)-1)
    if (t == 1)
      a = indexes[1]
    else
      if (indexes[t+1] != (indexes[t]+1))
        b = indexes[t]
        A = [A [a, b]]
        global a = indexes[t+1]
      end
    end
  end
  A = [A [a, indexes[end]]]
  A = A[:, 2:end]

  clear!(:a)
  return A
end



# Function: return time points and data w.p
# proportional to length of the time series (if ts_sample = true)
# otherwise it returns the longest time series.

function get_segment_ts(data, A, ts_sample)

  temp = [length(A[1, i]:A[2, i]) for i in 1:size(A)[2]]

  # - Sample time series w.p prop to its length.
  if ts_sample
    ind = sample(1:length(temp), Weights(temp))

  # - Sample longest time series.
  else
    ind = find(temp .== maximum(temp))
    if (length(ind) > 1) ind = sample(ind) end
  end

  y_ind = reshape(A[:, ind], 2)

  out = Dict("data" => data[y_ind[1]:y_ind[2]],
             "time" => y_ind[1]:y_ind[2])

  return out
end
