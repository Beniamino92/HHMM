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

# Function: function return design matrix with basis function,
#           that refers to all data points belonging to a certain
#           regime. Namely, time_star correspond to time points
#           that are related to the same regime.
function get_X_star(ω, time_star)

  M = length(ω)
  X = ones(length(time_star))

  for j in 1:M
    X = hcat(X, cos.(2π*time_star*ω[j]),sin.(2π*time_star*ω[j]))
  end

  return X[:, 2:end]
end

# # Function: function return design matrix with basis function.
# function get_X(ω, a, b)
#
#   M = length(ω)
#   time = a:b
#   X = ones(length(time))
#
#   for j in 1:M
#     X = hcat(X, cos.(2π*time*ω[j]),sin.(2π*time*ω[j]))
#   end
#   return X[:, 2:end]
# end

# Function: return detrended data via linear regression over time, and trend
function detrend(data, time)

  n = length(data)
  covariate_time = ones(n, 2)
  covariate_time[:, 2] = time
  b = inv(covariate_time'*covariate_time)*covariate_time'*data
  trend = covariate_time * b
  out = data - trend

  return out
end



# Function: log_posterior_ω
function log_posterior_ω_stationary(ω, β, time_star)
  X = get_X_star(ω, time_star)
  f = (-sum((y_star - X*β).^2)/(2*(σ^2)))[1]
  return f
end

# Function: log likelihood for β and ω
function log_likelik(β, ω, time_star)
  X = get_X_star(ω, time_star)
  out = -(n*log(2π))/2 -(n*log(σ^2))/2 - sum((y_star - X*β).^2)/(2*(σ^2))
  return out
end

# Function: log_posterior_beta
function log_posterior_β_stationary(β, ω, σ, σ_β, time_star)
  X = get_X_star(ω, time_star)
  f = (-sum((y_star - X*β).^2)/(2*(σ^2)) - ((β'*β)/(2*(σ_β^2))))[1]
  return f
end


# # Function: negative log_posterior_beta + auxiliary for optim
# function neg_log_posterior_β_stationary(β, ω, σ, σ_β, time_star)
#   X = get_X_star(ω, time_star)
#   f = (-sum((y_star - X*β).^2)/(2*(σ^2)) - ((β'*β)/(2*(σ_β^2))))[1]
#   return -f
# end
# function neg_f_posterior_β_stationary(β)
#   neg_log_posterior_β_stationary(β, ω, σ, σ_β, time_star)
# end

# # Function: negative gradient log posterior beta + auxiliary for optim
# function neg_grad_log_posterior_β_stationary(β, ω, σ, σ_β, time_star)
#
#   p = length(β)
#   g = zeros(p)
#   X = get_X_star(ω, time_star)
#
#   for i in 1:p
#     g[i] = sum((y_star - X*β).*X[:, i])/(σ^2) - (β[i]/(σ_β^2))
#   end
#
#   return -g
# end
# function neg_g_posterior_β_stationary!(storage, β)
#   storage[:] = neg_grad_log_posterior_β_stationary(β, ω, σ, σ_β, time_star)
# end


# # Function: negative hessian log posterior beta + auxiliary for optim
# function neg_hess_log_posterior_β_stationary(β, ω, σ, σ_β, time_star)
#
#   p = length(β)
#   h = zeros(p, p)
#   X = get_X_star(ω, time_star)
#
#   for i in 1:p
#     h[i, i] = -sum(X[:, i].^2)/(σ^2) - (1/(σ_β^2))
#   end
#
#   return -h
# end
# function neg_h_posterior_β_stationary!(storage, β)
#   storage[:, :] =  neg_hess_log_posterior_β_stationary(β, ω, σ, σ_β, time_star)
# end


# Function: within move. Proposed frequencies via mixture of FFT sampling and RW
#           (one-at-time frequency updating)
function within_model_move_stationary(info_segment_ts, m_current, β_current,
                                      ω_current, σ_current, time_star, λ, c, ϕ_ω, ψ_ω)

  if (2*length(ω_current) != (length(β_current))) error("dimension mismatch, ω and β") end

  global σ = σ_current
  time_indexes = copy(info_segment_ts["time"])



  # -------------------------- Sampling frequencies -------------------------


  period = periodogram(detrend(y, time_indexes))
  p = period.power
  p_norm = p ./sum(p)
  freq = period.freq

  U = rand()



  # ------------------ Gibbs step (FFT) ----------------

  if (U <= α_mixing)

    ω_current_aux = copy(ω_current)

    for j in 1:m_current
      ω_curr = copy(ω_current_aux)

      # Avoiding a vector with two same frequencies (D column would be linear dependent)
      aux_temp = false

      while (aux_temp == false)

        # Proposing frequencies
        global ω_star = sample(freq, Weights(p_norm))
        global ω_prop = copy(ω_curr)

        # Updating j-th component
        ω_prop[j] = ω_star

        if (! (any(vcat(ω_prop[1:(j-1)], ω_prop[(j+1):end]) .== ω_star)))
          aux_temp = true
        end

      end

      log_likelik_ratio = log_posterior_ω_stationary(ω_prop, β_current, time_star) -
                          log_posterior_ω_stationary(ω_curr, β_current, time_star)

      log_proposal_ratio = log(p_norm[searchsortedlast(freq, ω_curr[j])]) -
                           log(p_norm[searchsortedlast(freq, ω_star)])


      MH_ratio = exp(log_likelik_ratio + log_proposal_ratio)[1]

      U = rand()

      if (U <= min(1, MH_ratio))
        ω_current_aux = ω_prop
      else
        ω_current_aux = ω_curr
      end

      ω_out = sort(ω_current_aux)
    end

    ω_out = sort(ω_current_aux)



  # ------------------ Random Walk MH ------------------

  else

    ω_current_aux = copy(ω_current)

    for j in 1:m_current

      ω_curr = copy(ω_current_aux)
      aux_temp = false

      # Proposed frequency has to lie within [0,  0.5]
      while (aux_temp == false)
        global ω_star = rand(Normal(ω_current[j], σ_RW), 1)[1]
        if !(ω_star <= 0 || ω_star >= 0.5)
          aux_temp = true
        end
      end

      global ω_prop = copy(ω_curr)

      # Updating the j-th component
      ω_prop[j] = ω_star

      log_likelik_ratio = log_posterior_ω_stationary(ω_prop, β_current, time_star) -
                          log_posterior_ω_stationary(ω_curr, β_current, time_star)

      MH_ratio = exp.(log_likelik_ratio)[1]

      U = rand()

      if (U <= min(1, MH_ratio))
        ω_current_aux = ω_prop
      else
        ω_current_aux = ω_curr
      end
    end

    ω_out = sort(ω_current_aux)
  end





  #  -----------------  Sampling Basis Function Coefficients  -----------------


  X_post = get_X_star(ω_out, time_star)

  β_var_post = inv(eye(2*m_current)/(σ_β^2) + (X_post'*X_post)/(σ^2))
  β_var_post = (β_var_post' + β_var_post)/2
  β_mean_post = β_var_post*((X_post'*y_star)/(σ^2))

  β_out = rand(MultivariateNormal(β_mean_post, β_var_post), 1)


  #   ------- Sampling σ  --------

  X_post = get_X_star(ω_out, time_star)
  res_var = sum((y_star - X_post*β_out).^2)

  ν_post = (n + ν0)/2
  γ_post = (γ0 + res_var)/2

  σ_out  = sqrt.(rand(InverseGamma(ν_post, γ_post), 1))[1]



  #   -------

  output = Dict("β" => β_out, "ω" => ω_out, "σ" => σ_out)

  return output
end



# Function: birth_move (acceptance probability take account of sample freq to die (uniform)),
# and proposed frequencies are sorted.
function birth_move_stationary(info_segment_ts, m_current, β_current, ω_current,
                                σ_current, time_star, λ, c, ϕ_ω, ψ_ω)

  if (2*length(ω_current) != (length(β_current) )) error("dimension mismatch, ω and β") end

  global σ = σ_current

  time_indexes = info_segment_ts["time"]

  m_proposed = m_current + 1

  # - Proposing ω
  ω_current_aux = sort(vcat(0, ω_current, ϕ_ω))
  support_ω = Array{Vector{Float64}}(m_current + 1)
  for k in 1:(m_current + 1)
    support_ω[k] = [ω_current_aux[k] + ψ_ω, ω_current_aux[k+1] - ψ_ω]
  end
  length_support_ω = ϕ_ω - (2*(m_current + 1)*ψ_ω)

  ω_star = sample_uniform_continuous_intervals(1, support_ω)[1]
  ω_proposed = sort(vcat(ω_current, ω_star))

  # - Proposing β ∼ Normal(β̂_prop, Σ̂_prop)

  X_prop = get_X_star(ω_proposed, time_star)
  β_var_prop = inv(eye(2*m_proposed)/(σ_β^2) + (X_prop'*X_prop)/(σ^2))
  β_mean_prop = β_var_prop*((X_prop'*y_star)/(σ^2))
  β_proposed = rand(MvNormal(β_mean_prop, 0.5*(β_var_prop + β_var_prop')), 1)


  # - Obtaining β̂_curr, Σ̂_curr (for proposal ratio)
  X_curr = get_X_star(ω_current, time_star)
  β_var_curr = inv(eye(2*m_current)/(σ_β^2) + (X_curr'*X_curr )/(σ^2))
  β_mean_curr = β_var_curr*((X_curr'*y_star)/(σ^2))



  # ----- Evaluating acceptance probability

  # --- Log likelihood ratio
  log_likelik_prop = log_likelik(β_proposed, ω_proposed, time_star)
  log_likelik_curr = log_likelik(β_current, ω_current, time_star)
  log_likelik_ratio = log_likelik_prop - log_likelik_curr

  # --- Log prior ratio
  log_m_prior_ratio = log.(pdf(Poisson(λ), m_proposed)) - log(pdf(Poisson(λ), m_current))
  log_β_prior_ratio = (log.(pdf(MvNormal(zeros(2*m_proposed), (σ_β^2)*eye(2*m_proposed)) , β_proposed))[1]) -
                       log.(pdf(MvNormal(zeros(2*m_current), (σ_β^2)*eye(2*m_current)), β_current))
  log_ω_prior_ratio = log(2)

  log_prior_ratio = log_m_prior_ratio + log_β_prior_ratio +
                    log_ω_prior_ratio

  # --- Log proposal ratio
  log_proposal_β_prop = (-0.5*(β_proposed - β_mean_prop)'*inv(β_var_prop)*(β_proposed - β_mean_prop) -
                            log(sqrt(det(2π*β_var_prop))))[1]
  log_proposal_β_current = (-0.5*(β_current - β_mean_curr)'*inv(β_var_curr)*(β_current - β_mean_curr) -
                            log(sqrt(det(2π*β_var_curr))))[1]
  log_proposal_ω_proposed = log((1/length_support_ω))
  log_proposal_ω_current = log(1/m_proposed)

  log_proposal_birth_move = log(c*min(1, pdf(Poisson(λ), m_proposed)/pdf(Poisson(λ), m_current)))
  log_proposal_death_move = log(c*min(1, pdf(Poisson(λ), m_current)/pdf(Poisson(λ), m_proposed)))
  log_proposal_ratio = log_proposal_death_move - log_proposal_birth_move +
                       log_proposal_ω_current - log_proposal_ω_proposed + log_proposal_β_current -
                       log_proposal_β_prop



  # --- MH acceptance step
  MH_ratio_birth = log_likelik_ratio + log_prior_ratio + log_proposal_ratio
  epsilon_birth = min(1, exp(MH_ratio_birth))

  U = rand()

  if (U <= epsilon_birth)
    β_out = β_proposed
    ω_out = ω_proposed
    accepted = true
  else
    β_out = β_current
    ω_out = ω_current
    accepted = false
  end


  # -- Updating σ in a Gibb step

  X_post = get_X_star(ω_out, time_star)
  res_var = sum((y_star - X_post*β_out).^2)
  ν_post = (n + ν0)/2
  γ_post = (γ0 + res_var)/2

  σ_out = sqrt.(rand(InverseGamma(ν_post, γ_post), 1))[1]

  output = Dict("β" => β_out, "ω" => ω_out, "σ" => σ_out,
                "accepted" => accepted,
                "ω_star" => ω_star)
  return output
end


# Function: death_move (sample freq to die)
function death_move_stationary(info_segment_ts, m_current, β_current, ω_current,
                                σ_current, time_star, λ, c, ϕ_ω, ψ_ω)

  if (2*length(ω_current) != (length(β_current))) error("dimension mismatch, ω and β") end

  global σ = σ_current

  time_indexes = info_segment_ts["time"]

  m_proposed = m_current - 1

  index = sample(1:m_current)
  ω_proposed = vcat(ω_current[1:(index-1)], ω_current[(index+1):end])


  # - Proposing β ∼ Normal(β̂_prop, Σ̂_prop)

  X_prop = get_X_star(ω_proposed, time_star)
  β_var_prop = inv(eye(2*m_proposed)/(σ_β^2) + (X_prop'*X_prop)/(σ^2))
  β_mean_prop = β_var_prop*((X_prop'*y_star)/(σ^2))
  β_proposed = rand(MvNormal(β_mean_prop, 0.5*(β_var_prop + β_var_prop')), 1)


  # - Obtaining β̂_curr, Σ̂_curr (for proposal ratio)
  X_curr = get_X_star(ω_current, time_star)
  β_var_curr = inv(eye(2*m_current)/(σ_β^2) + (X_curr'*X_curr )/(σ^2))
  β_mean_curr = β_var_curr*((X_curr'*y_star)/(σ^2))


  length_support_ω = ϕ_ω - (2*(m_current)*ψ_ω)


  # ----- Evaluating acceptance probability

  # --- Log likelihood ratio
  log_likelik_prop = log_likelik(β_proposed, ω_proposed, time_star)
  log_likelik_curr = log_likelik(β_current, ω_current, time_star)
  log_likelik_ratio = log_likelik_prop - log_likelik_curr

  # --- Log prior ratio
  log_m_prior_ratio = log(pdf(Poisson(λ), m_proposed)) - log(pdf(Poisson(λ), m_current))
  log_β_prior_ratio = (log.(pdf(MvNormal(zeros(2*m_proposed), (σ_β^2)*eye(2*m_proposed)) , β_proposed))[1]) -
                       log.(pdf(MvNormal(zeros(2*m_current), (σ_β^2)*eye(2*m_current)), β_current))
  log_ω_prior_ratio = log(0.5)
  log_prior_ratio = log_m_prior_ratio + log_β_prior_ratio +
                    log_ω_prior_ratio

  # --- Log proposal ratio
  log_proposal_β_prop = (-0.5*(β_proposed - β_mean_prop)'*inv(β_var_prop)*(β_proposed - β_mean_prop) -
                            log(sqrt(det(2π*β_var_prop))))[1]
  log_proposal_β_current = (-0.5*(β_current - β_mean_curr)'*inv(β_var_curr)*(β_current - β_mean_curr) -
                            log(sqrt(det(2π*β_var_curr))))[1]
  log_proposal_ω_current = log((1/length_support_ω))
  log_proposal_ω_proposed = log(1/m_current)
  log_proposal_birth_move = log(c*min(1, pdf(Poisson(λ), m_current)/pdf(Poisson(λ), m_proposed)))
  log_proposal_death_move = log(c*min(1, pdf(Poisson(λ), m_proposed)/pdf(Poisson(λ), m_current)))

  log_proposal_ratio = log_proposal_birth_move - log_proposal_death_move +
                       log_proposal_ω_current - log_proposal_ω_proposed + log_proposal_β_current -
                       log_proposal_β_prop


  # --- MH acceptance step
  MH_ratio_death = log_likelik_ratio + log_prior_ratio + log_proposal_ratio
  epsilon_death = min(1, exp(MH_ratio_death))

  U = rand()
  if (U <= epsilon_death)
    β_out = β_proposed
    ω_out = ω_proposed
    accepted = true
  else
    β_out = β_current
    ω_out = ω_current
    accepted = false
  end


  # -- Updating σ in a Gibb step

  X_post = get_X_star(ω_out, time_star)
  res_var = sum((y_star - X_post*β_out).^2)
  ν_post = (n + ν0)/2
  γ_post = (γ0 + res_var)/2
  σ_out = sqrt.(rand(InverseGamma(ν_post, γ_post), 1))[1]


  output = Dict("β" => β_out, "ω" => ω_out, "σ" => σ_out,
                "accepted" => accepted)

  return output
end



# Function: Sampling uniformly from continuous disjoint subintervals
function sample_uniform_continuous_intervals(n_sample::Int64, intervals)

  out = zeros(n_sample)
  n_intervals = length(intervals)

  # Getting length of each interval
  len_intervals = zeros(n_intervals)
  for k in 1:n_intervals
      aux  = intervals[k][2]-intervals[k][1]
      if (aux < 0) aux = 0 end
      len_intervals[k] = aux
  end

  # Getting proportion of each interval
  weights = zeros(n_intervals)
  for k in 1:n_intervals
    weights[k] = len_intervals[k]/sum(len_intervals)
  end

  # Getting samples
  for j in 1:n_sample
    indicator = wsample(1:n_intervals, weights)
    out[j] = rand(Uniform(intervals[indicator][1], intervals[indicator][2]))
  end

  return out
end



# Function HR_RJMCMC for stationary time series.
# y is global, n is global
function RJMCMC_SegmentModelSearch(info_segment_ts, m_current, β_current,
                              ω_current, σ_current, time_star, λ, c, ϕ_ω,
                              ψ_ω, n_freq_max)

  if ( (length(β_current)) != (2*m_current) || (length(ω_current) != m_current) )
    error("dimension mismatch, ω and β") end

  # If m == 1, then either birth or within model move
  if (m_current == 1)

    birth_prob = c*min(1, (pdf(Poisson(λ), 2)/
                   pdf(Poisson(λ), 1)))
    U = rand()

    if (U <= birth_prob)
      MCMC = birth_move_stationary(info_segment_ts, m_current, β_current, ω_current, σ_current, time_star, λ, c, ϕ_ω, ψ_ω)
      m_out = m_current + Int64(MCMC["accepted"])
      β_out = MCMC["β"]
      ω_out = MCMC["ω"]
      σ_out = MCMC["σ"]

    else
      MCMC = within_model_move_stationary(info_segment_ts, m_current, β_current, ω_current, σ_current, time_star, λ, c, ϕ_ω, ψ_ω)
      m_out = m_current
      β_out = MCMC["β"]
      ω_out = MCMC["ω"]
      σ_out = MCMC["σ"]
    end

  # If m == n_freq_max, then either death or within model move
  elseif (m_current == n_freq_max)

    death_prob = c*min(1, (pdf(Poisson(λ), n_freq_max - 1)/
                   pdf(Poisson(λ), n_freq_max)))
    U = rand()
    if (U <= death_prob)
      MCMC = death_move_stationary(info_segment_ts, m_current, β_current, ω_current, σ_current, time_star, λ, c, ϕ_ω, ψ_ω)
      m_out = m_current - Int64(MCMC["accepted"])
      β_out = MCMC["β"]
      ω_out = MCMC["ω"]
      σ_out = MCMC["σ"]
    else
      MCMC = within_model_move_stationary(info_segment_ts, m_current, β_current, ω_current, σ_current, time_star, λ, c, ϕ_ω, ψ_ω)
      m_out = m_current
      β_out = MCMC["β"]
      ω_out = MCMC["ω"]
      σ_out = MCMC["σ"]
    end

  else

    birth_prob = c*min(1, (pdf(Poisson(λ), m_current + 1)/
                   pdf(Poisson(λ), m_current)))
    death_prob = c*min(1, (pdf(Poisson(λ), m_current - 1)/
                   pdf(Poisson(λ), m_current)))

    U = rand()

    # ----- Birth
    if (U <= birth_prob)
      MCMC = birth_move_stationary(info_segment_ts, m_current, β_current, ω_current, σ_current, time_star, λ, c, ϕ_ω, ψ_ω)
      m_out = m_current + Int64(MCMC["accepted"])
      β_out = MCMC["β"]
      ω_out = MCMC["ω"]
      σ_out = MCMC["σ"]

    # ----- Death
    elseif ((U > birth_prob) && (U <= (birth_prob + death_prob)))

      MCMC = death_move_stationary(info_segment_ts, m_current, β_current, ω_current, σ_current, time_star, λ, c, ϕ_ω, ψ_ω)
      m_out = m_current - Int64(MCMC["accepted"])
      β_out = MCMC["β"]
      ω_out = MCMC["ω"]
      σ_out = MCMC["σ"]

    # ---- Within model
    else
      MCMC = within_model_move_stationary(info_segment_ts, m_current, β_current, ω_current, σ_current,time_star, λ, c, ϕ_ω, ψ_ω)
      m_out = m_current
      β_out = MCMC["β"]
      ω_out = MCMC["ω"]
      σ_out = MCMC["σ"]
    end

  end

  return Dict("m" => m_out, "β" => β_out,
               "ω" => ω_out, "σ" => σ_out)
end
