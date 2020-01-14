cd("/Users/beniamino/Desktop/SpectralHMM/")
ENV["KMP_DUPLICATE_LIB_OK"]="TRUE"

path_SegmentModelSearch = string(pwd(),"/functions_RJMCMC_SegmentModelSearch.jl")
path_StickyHDPHMM = string(pwd(),"/functions_StickyHDPHMM.jl")
path_Auxiliary = string(pwd(),"/functions_auxiliary.jl")

include(path_SegmentModelSearch)
include(path_StickyHDPHMM)
include(path_Auxiliary)


# ------------ ILLUSTRATIVE EXAMPLE  ----------- #


# ---  Parameters Simulated Time Series -

T = 1450 # n obs

π_z_true = [[0.9900 0.0097 0.0003 ]
       [0.0001 0.9900 0.0099]
       [0.0097 0.0003 0.9900]] # transition probability matrix

π_init_true = [0.99, 0.005, 0.005] # initial distribution

m_true = [1, 1, 2] # n of freq
β_true = [[0.8, 0.8], # basis coefficients
          [0.2, 0.2],
          [1.0, 1.0, 1.0, 1.0]]
ω_true = [[1/25], # frequencies
          [1/19],
          [1/12, 1/8]]
σ_true = [0.4, 0.08, 0.3] # Gaussian innovations




# ---------------------- Parameters SpectralHMM -----------------

n_iter_MCMC = 15000    # n of MCMC iterations

Kz = 7               # maximum n of states
n_freq_max = 5        # maximum n of frequencies per state


# -- Hyperparametrs RJMCMC_SegmentModelSearch --

c_S = 0.4 # c for RJCMC - SegmentModelSearch
λ_S = 1  # λ for RJMCMC - SegmentModelSearch
ϕ_ω = 0.25 # New frequency is sampled from Unif(0, ψ_ω) - SegmentModelSearch
ψ_ω = (1/T) # Miminum distance between frequency - SegmentModelSearch
σ_β = 10 # prior β ∼ N(0, σ_β I )
α_mixing = 0.2 # mixing proportion for sampling ω (within-step)
σ_RW = (1/(50*T)) # variance RW for sampling ω (within-step)
ν0 = 1/100 # prior σ ~ InverseGamma(ν0/2, γ0)
γ0 = 1/100 #                 "


# --- Hyperparameters sticky HDP-HMM --

a_α = 1; b_α = 0.01 # - (γ+κ) ∼ Gamma(a_α,b_α)
a_γ = 1; b_γ = 0.01 # - γ ∼ Gamma(a_γ,b_γ)
c = 100; d = 1; # ρ ∼ Beta(c,d)
hyperHMMhyperparams = Dict("a_α" => a_α, "b_α" => b_α,
                           "a_γ" => a_γ, "b_γ" => b_γ,
                            "c" => c, "d" => d)

# --- Other parameteres --

n_inner_MCMC = 2        # n of inner RJMCMC for updating θ_j
n_min_obs = 10        # if a state is assigned less than n_min_obs, the
                      # parameters corresponding to that state are drawn
                      # from their priors.
sample_time_series = true # if true: when updating frequencies (sampling periodogram)
                          # for state j, select segment of time series with probability
                          # proportional to n of observation in that segment.
                          # if false: select MAP estimate.


# ------- Generating data -------

srand(77)

colors = ["blue", "red", "green"] # ------------  TO CHANGEEEEEE
z_true = generate_labels(π_z_true, π_init_true, T)
simulation = generate_data(z_true, β_true, ω_true, σ_true)
data = simulation["data"]
signal = simulation["signal"]

close(); plot_data_regime(data, z_true, colors)
plot(z_true)




# ------------------------ MCMC Objects -------------------------


# ---- Model Parameters
m_sample = ones(Int64, Kz, n_iter_MCMC + 1)
β_sample = zeros(Float64, Kz, 2*n_freq_max, n_iter_MCMC + 1)
ω_sample = zeros(Float64, Kz, n_freq_max, n_iter_MCMC + 1)
σ_sample = zeros(Float64, Kz, n_iter_MCMC + 1)

# ---- Mode Sequence
stateSeq_sample = zeros(Int64, T, n_iter_MCMC + 1)
indSeq_sample = zeros(Int64, T, Kz, n_iter_MCMC + 1)
totSeq_sample = zeros(Int64, Kz, n_iter_MCMC + 1)
classProb_sample = zeros(Float64, n_iter_MCMC + 1, T, Kz) # necessary for postprocessing
#                                                           the sample with relabelling
#                                                           algorithm.

# ---- Transition Distributions (π) & Overall Dish Rating (β)
π_z_sample = zeros(Float64, Kz, Kz, n_iter_MCMC + 1)
π_init_sample = zeros(Float64, Kz, n_iter_MCMC + 1)
α_vec_sample = zeros(Float64, Kz, n_iter_MCMC + 1)

# ---- HMM Hyperparameters
η_plus_κ_sample = zeros(Float64, n_iter_MCMC + 1)
γ_sample = zeros(Float64, n_iter_MCMC + 1)
ρ_sample = zeros(Float64, n_iter_MCMC + 1)

# ---- State Counts:
N_sample = zeros(Int64, Kz+1, Kz, n_iter_MCMC+1) #  N(i,j) = # z_t = i to z_{t+1}=j transitions in z_{1:T}. N(Kz+1,i) = 1 for i=z_1.
M_sample = zeros(Int64, Kz+1, Kz, n_iter_MCMC+1) #  M(i,j) = # of tables in restaurant i serving dish k
barM_sample = zeros(Int64, Kz+1,Kz, n_iter_MCMC+1)  # barM(i,j) = # of tables in restaurant i considering dish k
sum_w_sample = zeros(Int64, Kz, n_iter_MCMC+1) #  sum_w(i) = # of overriden dish assignments in restaurant i

# ---- Log Likelihood
log_likelik_sample = zeros(Float64, n_iter_MCMC+1)




# --------- Starting Values MCMC objects ----------

# -- Concentration Parameters
HMMhyperparms = sample_hyperparams_init(hyperHMMhyperparams)
η_plus_κ_sample[1] = HMMhyperparms["α_plus_κ"]
γ_sample[1] = HMMhyperparms["γ"]
ρ_sample[1] = HMMhyperparms["ρ"]

# -- Sample transition distributions π_z, initial distribution π_init
#    and global transition distribution β.
dist_struct = sample_dist(η_plus_κ_sample[1], γ_sample[1], ρ_sample[1],
                          N_sample[:, :, 1], barM_sample[:, :, 1])
π_z_sample[:, :, 1] = dist_struct["π_z"]
# z_start = vcat(fill(2, Int64(T/2)),
#                fill(5, Int64(T/2)))
z_start = fill(2, T)
π_init_start = dist_struct["π_init"]
π_init_sample[:, 1] = copy(π_init_start)
α_vec_sample[:, 1] = dist_struct["β_vec"]


# -- Emission parameters
σ_smoothing = 0.1
get_starting_value_θ(z_start, 3)
m_sample[:, 1]
β_sample[:, :, 1]
ω_sample[:, :, 1]
σ_sample[:, 1]

smooth_period = periodogram(detrend(data, 1:T), window = gaussian(T, σ_smoothing))
I_smooth = smooth_period.power
freq = smooth_period.freq
close()
plot(freq, I_smooth)



srand(10)
# ------------------ Spectral HDP-HMM  -------------------

@showprogress for tt in 2:(n_iter_MCMC+1)


  #  --- Block sample z_{1:T} | y_{1:T}
  modes_struct = sample_z(π_z_sample[:, :, tt-1], π_init_sample[:, tt-1],
                         m_sample[:, tt-1], β_sample[:, :, tt-1],
                         ω_sample[:, :, tt-1], σ_sample[:, tt-1])
  N_sample[:, :, tt] = modes_struct["N"]
  stateSeq_sample[:, tt] = modes_struct["stateSeq"]
  classProb_sample[tt, :, :] = modes_struct["classProb"] # need for relabelling algorithm

  totSeq = copy(modes_struct["totSeq"])
  indSeq = copy(modes_struct["indSeq"])

  indSeq_sample[:, :, tt] = copy(indSeq)
  totSeq_sample[:, tt] = copy(totSeq)

  if (tt % 1000 == 0)
     println(tt, " - totSeq - : ", totSeq)
     #close(); plot(data); plot(stateSeq_sample[:, tt], color = "red")
  end


  # --- Based on mode sequence assignment, sample how many tables in each
  #     restaurant are serving each of the selected dishes. Also sample the
  #     dish override variables:

  tables = sample_tables(N_sample[:, :, tt], η_plus_κ_sample[tt-1],
                         ρ_sample[tt-1], α_vec_sample[:, tt-1])
  M_sample[:, :, tt] = tables["M"]
  barM_sample[:, :, tt] = tables["barM"]
  sum_w_sample[:, tt] = tables["sum_w"]

  # --- Sample the transition distributions π_z, initial distribution
  #     π_init, and avg transition distribution beta:
  dist_struct = sample_dist(η_plus_κ_sample[tt-1], γ_sample[tt-1],
                            ρ_sample[tt-1],
                            N_sample[:, :, tt], barM_sample[:, :, tt])
  π_z_sample[:, :, tt] = dist_struct["π_z"]
  π_init_sample[:, tt] = dist_struct["π_init"]
  α_vec_sample[:, tt] = dist_struct["β_vec"]


  # ----------- Updating Emission Parameters
  for j in 1:Kz

      # -- Updates for modes with at least n_min observations
      if (totSeq[j] > n_min_obs)
        temp_ind = copy(indSeq[1:totSeq[j], j])
        temp_ind_seg = get_time_indexes(temp_ind)
        len_seg = [length(temp_ind_seg[1, ii]:temp_ind_seg[2, ii])
          for ii in 1:size(temp_ind_seg)[2]]


        info_segment_ts = []
        if (any(len_seg .>= n_min_obs))

          #  -- Sample segment until it has n_min_obs observations
          while true
            info_segment_ts = get_segment_ts(data, temp_ind_seg, sample_time_series)
            if (length(info_segment_ts["time"]) >= n_min_obs) break end
          end


          global y = info_segment_ts["data"]
          global time_star = indSeq[1:totSeq[j], j]
          global y_star = data[time_star]
          global n = length(y_star)
          # global σ_RW = (1/(50*n))

          m_current = m_sample[j, tt-1]
          β_current = β_sample[j, 1:(2*m_current), tt-1]
          ω_current = ω_sample[j, 1:m_current, tt-1]
          σ_current = σ_sample[j, tt-1]

          m_inner = zeros(Int64, n_inner_MCMC+1)
          β_inner = zeros(Float64, 2*n_freq_max, n_inner_MCMC+1)
          ω_inner = zeros(Float64, n_freq_max, n_inner_MCMC+1)
          σ_inner = zeros(Float64, n_inner_MCMC+1)

          m_inner[1] = copy(m_current)
          β_inner[1:(2*m_inner[1]), 1] = copy(β_current)
          ω_inner[1:m_inner[1], 1] = copy(ω_current)
          σ_inner[1] = copy(σ_current)

          # Inner loop, updating emission parameters
          for ii in 2:(n_inner_MCMC+1)

              m_temp = copy(m_inner[ii-1])
              β_temp = copy(β_inner[1:(2*m_temp), ii-1])
              ω_temp = copy(ω_inner[1:m_temp, ii-1])
              σ_temp = copy(σ_inner[ii-1])

              # -- Segment Model Move --
              MCMC = RJMCMC_SegmentModelSearch(info_segment_ts, m_temp, β_temp,
                              ω_temp, σ_temp, time_star, λ_S, c_S, ϕ_ω, ψ_ω, n_freq_max)

              m_inner[ii] = MCMC["m"]
              β_inner[1:(2*m_inner[ii]), ii] = MCMC["β"]
              ω_inner[1:m_inner[ii], ii] = MCMC["ω"]
              σ_inner[ii] = MCMC["σ"]

          end

          m_sample[j, tt] = copy(m_inner[end])
          β_sample[j, 1:(2*m_sample[j, tt]), tt] = copy(β_inner[1:(2*m_inner[end]), end])
          ω_sample[j, 1:m_sample[j, tt], tt] = copy(ω_inner[1:m_inner[end], end])
          σ_sample[j, tt] = copy(σ_inner[end])

        else
          m_sample[j, tt] = 1
          β_sample[j, 1:(2*m_sample[j, tt]), tt] = vcat(rand(MultivariateNormal(zeros((2*m_sample[j, tt])),
                                                      eye((2*m_sample[j, tt])))))
          ω_sample[j, 1:m_sample[j, tt], tt] = rand(Uniform(0, ϕ_ω))
          σ_sample[j, tt] = rand(InverseGamma(ν0/2, γ0/2))
        end

      else
          m_sample[j, tt] = 1
          β_sample[j, 1:(2*m_sample[j, tt]), tt] = vcat(rand(MultivariateNormal(zeros((2*m_sample[j, tt])),
                                                      eye((2*m_sample[j, tt])))))
          ω_sample[j, 1:m_sample[j, tt], tt] = rand(Uniform(0, ϕ_ω))
          σ_sample[j, tt] = rand(InverseGamma(ν0/2, γ0/2))
      end
  end

  # --- Resample concentration parameters:
  HMMhyperparams = sample_hyperparams(N_sample[:, :, tt], M_sample[:, :, tt],
                                      barM_sample[:, :, tt], sum_w_sample[:, tt],
                                      η_plus_κ_sample[tt-1], γ_sample[tt-1],
                                      hyperHMMhyperparams)
  η_plus_κ_sample[tt] = HMMhyperparams["α_plus_κ"]
  γ_sample[tt] = HMMhyperparams["γ"]
  ρ_sample[tt] = HMMhyperparams["ρ"]


  # --- Log likelihood

  z = copy(stateSeq_sample[:, tt])
  m = copy(m_sample[:, tt])
  ω = copy(ω_sample[:, :, tt])
  β = copy(β_sample[:, :, tt])
  σ = copy(σ_sample[:, tt])
  π_z = copy(π_z_sample[:, :, tt])
  π_init = copy(π_init_sample[:, tt])

  log_likelik_sample[tt] = get_likelik_system(z, m, ω, β, σ, π_z, π_init)
end

path_MCMC_simulation = string(pwd(),"/simul.jld")

# save(path_MCMC_simulation, "m", m_sample, "β", β_sample, "ω", ω_sample,
#      "σ", σ_sample, "indSeq", indSeq_sample, "totSeq", totSeq_sample,
#       "stateSeq", stateSeq_sample, "classProb", classProb_sample,
#       "π_z", π_z_sample, "π_init", π_init_sample, "β_vec", α_vec_sample,
#       "α_plus_κ", η_plus_κ_sample, "γ", γ_sample, "ρ", ρ_sample,
#       "N", N_sample, "M", M_sample, "barM", barM_sample,
#       "sum_w", sum_w_sample, "log_likelik", log_likelik_sample, "data", data)


MCMC_simul = load(path_MCMC_simulation)
data = MCMC_simul["data"]
T = length(data)


# ---- Model Parameters
m_sample = MCMC_simul["m"]
β_sample = MCMC_simul["β"]
ω_sample = MCMC_simul["ω"]
σ_sample = MCMC_simul["σ"]

# ---- Mode Sequence
stateSeq_sample = MCMC_simul["stateSeq"]
indSeq_sample = MCMC_simul["indSeq"]
totSeq_sample = MCMC_simul["totSeq"]
classProb_sample = MCMC_simul["classProb"]

# ---- Transition Distributions (π) & Overall Dish Rating (β)
π_z_sample = MCMC_simul["π_z"]
π_init_sample = MCMC_simul["π_init"]
β_vec_sample = MCMC_simul["β_vec"]

# ---- HMM Hyperparameters
η_plus_κ_sample = MCMC_simul["α_plus_κ"]
γ_sample = MCMC_simul["γ"]
ρ_sample = MCMC_simul["ρ"]

# ---- State Counts:
N_sample = MCMC_simul["N"] #  N(i,j) = # z_t = i to z_{t+1}=j transitions in z_{1:T}. N(Kz+1,i) = 1 for i=z_1.
M_sample = MCMC_simul["M"] #  M(i,j) = # of tables in restaurant i serving dish k
barM_sample = MCMC_simul["barM"]  # barM(i,j) = # of tables in restaurant i considering dish k
sum_w_sample = MCMC_simul["sum_w"] #

# --- Log Likelikhood:
log_likelik_sample = MCMC_simul["log_likelik"]

n_iter_MCMC = size(stateSeq_sample)[2] - 1
Kz = size(m_sample)[1]




# -------------- Diagnostic Convergence ----------------

close(); plot(log_likelik_sample)
burn_in_MCMC = Int64(0.5*n_iter_MCMC)


# -  Burned Chains

m_final = m_sample[:, burn_in_MCMC:end]
β_final = β_sample[:, :, burn_in_MCMC:end]
ω_final = ω_sample[:, :, burn_in_MCMC:end]
σ_final = σ_sample[:, burn_in_MCMC:end]
n_final = size(σ_final)[2]

totSeq_final = totSeq_sample[:, burn_in_MCMC:end]
stateSeq_final = stateSeq_sample[:, burn_in_MCMC:end]
indSeq_final = indSeq_sample[:, :, burn_in_MCMC:end]
classProb_final = classProb_sample[burn_in_MCMC:end, :, :]

η_plus_κ_final = η_plus_κ_sample[burn_in_MCMC:end]
γ_final = γ_sample[burn_in_MCMC:end]
ρ_final = ρ_sample[burn_in_MCMC:end]
n_final = length(ρ_final)

N_final = N_sample[:, :, burn_in_MCMC:end]
π_z_final = π_z_sample[:, :, burn_in_MCMC:end]
π_init_final = π_init_sample[:, burn_in_MCMC:end]


# - Posterior probability for the number of unique states:
n_unique_regimes = [length(unique(stateSeq_final[:, ii]))
                          for ii in 1:n_final]
for i = 1:Kz
    prob = length(find(n_unique_regimes .== i))/n_final
    println(" % n. of regime = ", i, ": ", prob)
end

# - Selecting  MCMC iterations for whithin model posterior analysis
n_regime_est = mode(n_unique_regimes)
indexes_analysis = find(n_unique_regimes .== n_regime_est)

z = stateSeq_final[:, indexes_analysis]'
p = classProb_final[indexes_analysis, :, :]


R"""
library("label.switching")
label_sim = label.switching(method = "STEPHENS", z = $z,
                            K = $Kz, p = $p)
"""

label_sim = rcopy(R"label_sim")
stateSeq_est = reshape(convert(Array{Int64},label_sim[:clusters]), T)
permuted_labels = label_sim[:permutations][:STEPHENS]'






# -- Conditioning MCMC sample to modal number of states,
#    for within model analysis.
m_analysis = m_final[:, indexes_analysis]
ω_analysis = ω_final[:, :, indexes_analysis]
β_analysis = β_final[:, :, indexes_analysis]
σ_analysis = σ_final[:, indexes_analysis]
π_z_analysis = π_z_final[:, :, indexes_analysis]
π_init_analysis = π_init_final[:, indexes_analysis]

#unique_regimes_analysis = unique_regimes_final[:, indexes_analysis]
stateSeq_analysis = stateSeq_final[:, indexes_analysis]


z_unique = unique(stateSeq_analysis)
state = z_unique[3] # for state in z_unique

get_summary_permuted(state, permuted_labels; plotting = true)
