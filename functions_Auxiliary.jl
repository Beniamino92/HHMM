using PyPlot
using Distributions
using ProgressMeter
using StatsBase
using Optim
using JLD
using DSP
using RCall
using CSV
using DataFrames





# Function: analtical formulation spectrum AR(p)
function arspec(ω, ϕ, σ)

    n_ω = length(ω)
    out = zeros(Float64, n_ω)
    p = length(ϕ)

    for i in 1:n_ω
        temp = 0.0
        for j in 1:p
            temp = temp + ϕ[j] * (exp(-im*(2π)*ω[i])^j)
        end
        out[i] = (σ^2)/(abs((1- temp))^2)
    end

    return out
end

# Function: generate state prediction for (n_pred) observations
function generate_state_prediction(n_pred)

    π_z = copy(π_z_est_tot)
    Kz = size(π_z, 2)
    z_end = z_est[end]

    z_pred = []

    for tt in (T+1):(T+n_pred)
        if (tt == T + 1)
            z_weights = Weights(π_z[z_end, states_analysis])
        else
            z_weights = Weights(π_z[z_pred[end], states_analysis])
        end
        append!(z_pred, sample(states_analysis, z_weights))
    end

    return z_pred
end


# Function: given z, returns info about n .of observations for states
#           in all segments determined by z.
function get_z_info(z)

  z_info = zeros(Int64, 2)
  count = 1

  for t in 2:T
    if (z[t] != z[t-1])
      z_info = hcat(z_info, [z[t-1], count])
      count = 0
    end

    count+=1

  end
  z_info = hcat(z_info, [z[end], count])

  return(z_info[:, 2:end])
end



# Function: simulate two_state AR_HMM
function simulate_AR_HMM(ψ_AR, z_true; warm_up = 100)

  data = zeros(T+warm_up)
  z = vcat(z_true[1:warm_up], z_true)

  for t in 3:(T+warm_up)
    data[t] = ψ_AR[z[t]][:ψ_1]*data[t-1] + ψ_AR[z[t]][:ψ_2]*data[t-2] +
              rand(Normal(0, σ_AR[z[t]]))
  end

  return(data[(warm_up+1):end])
end




# Function: Generate Data
function generate_data(z, β, ω, σ)

  T = length(z)

  locations_CP = get_info_changepoints(z)["locations"]
  n_CP = get_info_changepoints(z)["n_CP"]
  temp = vcat(1, locations_CP, T)

  signal, noise = [],[]

  for j in 1:(length(temp)-1)

    if (j==1)
      a = temp[j]
      b = temp[j+1]
    else
      a = temp[j]+1
      b = temp[j+1]
    end

    regime = z[temp[j]+1]

    X = get_X(ω[regime], a, b)
    f = X*vcat(β[regime])
    ε = rand(Normal(0, σ[regime]), size(X)[1])

    append!(signal, f)
    append!(noise, ε)
  end

  data = signal + noise
  out = Dict("data" => data, "signal" => signal)

  return out
end




# Function: get number of change-points and their locations, given a vector
#           of modes z.
function get_info_changepoints(z)

    n_CP = 0
    locations = []

    for t in 1:(T-1)
        if (z[t] != z[t+1])
            n_CP = n_CP + 1
            append!(locations, t)
        end
    end

    output = Dict("n_CP" => n_CP,
                  "locations" => locations)
    return output
end

function get_marginal_ω(sample, seg, m, m_seg, indexes_final, unique_regimes_final)

  aux = m_seg[indexes_final]
  indexes_inner = find(aux .== mode(m_seg))

  new_sample = zeros(Float64, length(indexes_inner))

  temp_sample = (sample[:, m, indexes_final])[:, indexes_inner]
  temp_z = (unique_regimes_final[seg, indexes_final])[indexes_inner]
  for ii in 1:length(indexes_inner)
    new_sample[ii] = temp_sample[temp_z[ii], ii]
  end

  return new_sample
end

# Function:  given time_indexes and corresponding
#            latent variable z_t's, it tells me at which regime time_in-
#             dexes belong to.
function get_regime(time, z)

  if (length(unique(z[time])) != 1) error("Time points correspond to more than one segment") end

  return unique(z[time])[1]
end



# Plot the data with different color for each regime.
function plot_data_regime(data, z, colors, lw = 1)


  # if (length(colors) != Kz)
  #   error("Dimension Mismatch: colors and Kz")
  # end

  segment_locations = vcat(1, get_info_changepoints(z)["locations"], T)


  for ii in 1:1:(length(segment_locations)-1)
    if (ii == 1)
      end_points = segment_locations[ii], segment_locations[ii+1]
    else
      end_points = segment_locations[ii] + 1, segment_locations[ii+1]
    end

    data_seg = data[end_points[1]:end_points[2]]
    regime = z[end_points[1]]
    plot(end_points[1]:end_points[2], data_seg, color = colors[regime], linewidth = lw)
  end
end

function plot_scatter_regime(data, z, colors, s = 3)


  # if (length(colors) != Kz)
  #   error("Dimension Mismatch: colors and Kz")
  # end

  segment_locations = vcat(1, get_info_changepoints(z)["locations"], T)


  for ii in 1:1:(length(segment_locations)-1)
    if (ii == 1)
      end_points = segment_locations[ii], segment_locations[ii+1]
    else
      end_points = segment_locations[ii] + 1, segment_locations[ii+1]
    end

    data_seg = data[end_points[1]:end_points[2]]
    regime = z[end_points[1]]
    scatter(end_points[1]:end_points[2], data_seg, color = colors[regime], s = s)
  end
end


# Function: compute estimated signal
function get_estimated_signal(indexes_final)

  y_estimates = zeros(Float64, length(indexes_final), T)

  aux = 1
  for tt in indexes_final

    stateSeq = stateSeq_final[:, tt]
    segment_locations = vcat(1, get_info_changepoints(stateSeq)["locations"], T)

    y_temp = []

    for ii in 1:(length(segment_locations)-1)

      if (ii == 1)
        end_points = segment_locations[ii], segment_locations[ii+1]
      else
        end_points = segment_locations[ii] + 1, segment_locations[ii+1]
      end

      regime = stateSeq[end_points[1]]
      m = m_final[regime, tt]
      X = get_X(ω_final[regime, 1:m, tt], end_points[1], end_points[2])
      β = β_final[regime, 1:(2*m), tt]

      append!(y_temp, X*β)
    end

    y_estimates[aux, :] = y_temp
    aux += 1
  end

  y_final_fit = reshape(mean(y_estimates, 1), T)

  out = Dict("signal_mean" => y_final_fit,
             "signal_sample" => y_estimates)

  return out
end

# Function: returns (stable) indexes for MCMC diagnostic.
function get_indexes_analysis(locations, s_mean_vec, n_CP_est, indexes_final)

    output = []
    for tt in indexes_final

      loc_temp = locations[1:n_CP_est, tt]

      global jj = 1
      temp = true
      while (temp == true && jj < n_CP_est)

        s_est = loc_temp[jj]
        s_mean = s_mean_vec[jj]

        if (s_est < (s_mean - delta) || s_est > (s_mean + delta))
          temp = false
        end

        jj = jj + 1
      end

      if (temp) append!(output, tt) end

    end
    return output
end

# Function: compute estimated signal
function get_estimated_signal(indexes_final)

  y_estimates = zeros(Float64, length(indexes_final), T)

  aux = 1
  for tt in indexes_final

    stateSeq = stateSeq_final[:, tt]
    segment_locations = vcat(1, get_info_changepoints(stateSeq)["locations"], T)

    y_temp = []

    for ii in 1:(length(segment_locations)-1)

      if (ii == 1)
        end_points = segment_locations[ii], segment_locations[ii+1]
      else
        end_points = segment_locations[ii] + 1, segment_locations[ii+1]
      end

      regime = stateSeq[end_points[1]]
      m = m_final[regime, tt]
      X = get_X(ω_final[regime, 1:m, tt], end_points[1], end_points[2])
      β = β_final[regime, 1:(2*m), tt]

      append!(y_temp, X*β)
    end

    y_estimates[aux, :] = y_temp
    aux += 1
  end

  y_final_fit = reshape(mean(y_estimates, 1), T)

  out = Dict("signal_mean" => y_final_fit,
             "signal_sample" => y_estimates)

  return out
end

# Generate hidden state sequence, given transition matrix π_z
# inital distribution π_init, and number of observations T
function generate_labels(π_z, π_init, T)

  n_states = size(π_z, 2)

  labels = zeros(Int64, T)

  for t in 1:T
    if (t == 1)
      labels[t] = sample(1:n_states, Weights(π_init))
    else
      labels[t] = sample(1:n_states, Weights(π_z[labels[t-1], :]))
    end
  end

  return labels
end

# Function: obtain summary of frequency location and
# corresponding power (posterior mean and std), without
# permuting the posterior sample.
function get_summary(regime; plotting = true)

  seg = find(unique_regimes_analysis[:, 1] .== regime)[1]

  m_seg = [m_analysis[unique_regimes_analysis[seg, i], i] for i in 1:length(indexes_analysis)]
  temp = unique_regimes_analysis[seg, :]
  m_est = mode(m_seg)
  modal_indexes = intersect(find(temp .== mode(temp)), find(m_seg .== m_est))

  if plotting == true
    close();
    for m in 1:m_est
      new_sample = [ω_analysis[temp[i], m, i] for i in modal_indexes]
      subplot(m_est, 1, m)
      new_sample_no_outliers = new_sample[new_sample .< (mean(new_sample) + 0.01)]
      plot(new_sample_no_outliers)
    end
  end

  β_aux = zeros(Float64, 2*m_est, length(modal_indexes))
  σ_aux = zeros(Float64, length(modal_indexes))
  for i in 1:length(modal_indexes)
      β_aux[:, i] = β_analysis[temp[i], 1:(2*m_est), i]
      σ_aux[i] = σ_analysis[temp[i], i]
  end

  summary_ω = zeros(m_est, 2)
  summary_β = zeros(2*m_est)
  summary_power = zeros(m_est, 2)
  summary_σ = mean(σ_aux)


  for m in 1:m_est

    new_sample_ω = [ω_analysis[temp[i], m, i] for i in modal_indexes]
    new_sample_power = [sqrt(sum(β_analysis[temp[i], [2*m-1, 2*m], i].^2)) for i in modal_indexes]

    indexes_no_outliers = new_sample_ω .< (mean(new_sample_ω) + 0.01)
    new_sample_ω_no_outliers = new_sample_ω[indexes_no_outliers]
    new_sample_power_no_outliers = new_sample_power[indexes_no_outliers]

    summary_ω[m, 1] = mean(new_sample_ω_no_outliers)
    summary_ω[m, 2] = sqrt(var(new_sample_ω_no_outliers))
    summary_power[m, 1] = mean(new_sample_power_no_outliers)
    summary_power[m, 2] = sqrt(var(new_sample_power_no_outliers))
    summary_β[(2*m-1):(2*m)] = mean(β_aux[(2*m-1):(2*m), indexes_no_outliers], 2)

  end


  return Dict(:freq => summary_ω, :β => summary_β,
              :σ => summary_σ, :power => summary_power)
end

# Function: given permuted labels and regime j,
#           obtain within-model summary statistics
#           of posterior frequencies and power,
#           (conditional on modal n.of.regimes and n.of. freq per regime).
#           If plotting == true, plot trace for frequencies.
function get_summary_permuted(j, permuted_labels; plotting = true)

    n_simul = size(permuted_labels, 2)

    perm = permuted_labels[j, :]
    m_j = [m_analysis[perm[ii], ii] for ii in 1:n_simul]
    m_est = mode(m_j)
    modal_indexes = find(m_j .== m_est)

    summary_ω = zeros(m_est, 2)
    summary_power = zeros(m_est, 2)
    summary_σ = zeros(2)

    σ_trace = [σ_analysis[perm[ii], ii] for ii in modal_indexes]
    summary_σ[1] = mean(σ_trace)
    summary_σ[2] = sqrt(var(σ_trace))

    close();
    for m in 1:m_est

        ω_trace = [ω_analysis[perm[ii], m, ii] for ii in modal_indexes]
        summary_ω[m, 1] = mean(ω_trace)
        summary_ω[m, 2] = sqrt(var(ω_trace))

        power_trace = [sqrt(sum(β_analysis[perm[ii], [2*m-1, 2*m], ii].^2))
                      for ii in modal_indexes]
        summary_power[m, 1] = mean(power_trace)
        summary_power[m, 2] = sqrt(var(power_trace))

        if (plotting == true)
            subplot(m_est, 1, m)
            plot(ω_trace)
            suptitle("j = $j", fontsize = 15)
        end
    end

    idx_aux = find(summary_power[:, 1] .== maximum(summary_power[:, 1]))[1]
    ω_dominant = summary_ω[idx_aux, 1]


    return Dict(:freq => summary_ω, :power => summary_power,
                :ω_dominant => ω_dominant)
end


# Function: get_time_varying_ω (averaged over MCMC iterations)
function get_time_varying_ω()

    m_post = m_final[:, indexes_analysis]
    ω_post = ω_final[:, :, indexes_analysis]
    β_post = β_final[:, :, indexes_analysis]
    z_post = stateSeq_final[:, indexes_analysis]

    n_unique_states_post = n_uniqe_regimes_final[indexes_analysis]
    unique_states_post = unique_regimes_final[:, indexes_analysis]

    time_varying_ω_sample = zeros(Float64, T, length(indexes_analysis))

    for tt in 1:length(indexes_analysis)

        z = z_post[:, tt]
        K = n_unique_states_post[tt]
        active_states = unique_states_post[1:K, tt]
        dict = Dict()

        for regime in active_states

            n_freq = m_post[regime, tt]
            ω = ω_post[regime, 1:n_freq, tt]
            β = β_post[regime, 1:2*n_freq, tt]

            power = [sqrt(sum(β[2*m-1:2*m].^2)) for m in 1:n_freq]
            ω_dominant = ω[find(power .== maximum(power))[1]]
            dict[regime] = ω_dominant

        end

        time_varying_ω_sample[:, tt] = [dict[z[ii]] for ii in 1:T]
    end


    time_varying_ω_mean = mean(time_varying_ω_sample, 2)

    return Dict(:mean => time_varying_ω_mean, :sample => time_varying_ω_sample)
end

function get_time_varying_ω_permuted(stateSeq_est, permuted_labels)


    active_states = unique(stateSeq_est)
    time_varying_peak_est = zeros(Float64, T)
    dict = Dict()


    for regime in active_states
        summary_regime = get_summary_permuted(regime, permuted_labels; plotting = false)
        dict[regime] = summary_regime
    end

    for t in 1:T
        z = stateSeq_est[t]
        time_varying_peak_est[t] = dict[z][:ω_dominant]
    end

    return time_varying_peak_est
end

# Function: generating signal prediction + 5% Gaussian C.I
function generate_signal_prediction(z_pred)

    signal_pred = []
    upper = []
    lower = []


    n_pred = length(z_pred)
    z_aux = vcat(z_est, z_pred)

    for tt in (T+1):(T+n_pred)

        regime = z_aux[tt]
        ω = Dict_summary[regime][:freq][:, 1]

        β = Dict_summary[regime][:β]
        σ = Dict_summary[regime][:σ]
        interval_975 = quantile(Normal(0, σ), .975)

        x_t = get_X(ω, tt, tt)
        signal_t = (x_t * β)[1]
        upper_t = signal_t + interval_975
        lower_t = signal_t - interval_975
        append!(signal_pred, signal_t)
        append!(upper, upper_t)
        append!(lower, lower_t)

    end


    return Dict(:signal => convert(Array{Float64}, signal_pred),
                :upper => convert(Array{Float64}, upper),
                :lower => convert(Array{Float64}, lower))
end


# Super inefficient function to get the likelihood of the system.
function get_likelik_system(z, m, ω, β, σ, π_z, π_init)

  tot = 0.0

  Threads.@threads for ii in 1:T

    X = get_X_star(ω[z[ii], 1:m[z[ii]]], ii)
    β_aux = β[z[ii], 1:(2*m[z[ii]])]
    μ = (X*β_aux)[1]

    likelik = pdf(Normal(μ, σ[z[ii]]), data[ii])

    if ii == 1
      transition_prob = π_init[z[ii]]
    else
      transition_prob = π_z[z[ii-1], z[ii]]
    end

    tot +=  log(transition_prob) + log(likelik)
  end

  return tot
end
