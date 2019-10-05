# Function: function return design matrix with basis function.
function get_X(ω, a, b)

  M = length(ω)
  time = a:b
  X = ones(length(time))

  for j in 1:M
    X = hcat(X, cos.(2π*time*ω[j]),sin.(2π*time*ω[j]))
  end
  return X[:, 2:end]
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
# corresponding power (posterior mean and std)
function get_summary(regime; plot = true)

  seg = find(unique_regimes_analysis[:, 1] .== regime)[1]

  m_seg = [m_analysis[unique_regimes_analysis[seg, i], i] for i in 1:length(indexes_analysis)]
  temp = unique_regimes_analysis[seg, :]
  m_est = mode(m_seg)
  modal_indexes = intersect(find(temp .== mode(temp)), find(m_seg .== m_est))

  if plot == true
    close();
    for m in 1:m_est
      new_sample = [ω_analysis[temp[i], m, i] for i in modal_indexes]
      subplot(m_est, 1, m)
      new_sample_no_outliers = new_sample[new_sample .< (mean(new_sample) + 0.01)]
      PyPlot.plot(new_sample_no_outliers)
    end
  end

  # if plot
  #   close();
  #   for m in 1:m_est
  #     new_sample = [ω_analysis[temp[i], m, i] for i in modal_indexes]
  #     subplot(m_est, 1, m)
  #     new_sample_no_outliers = new_sample[new_sample .< (mean(new_sample) + 0.01)]
  #     plot(new_sample_no_outliers)
  #   end
  # end

  summary_ω = zeros(m_est, 2)
  summary_power = zeros(m_est, 2)


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
  end


  return Dict(:freq => summary_ω, :power => summary_power)
end
