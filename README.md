
# HHMM (Harmonic Hidden Markov Model)

HHMM is a Julia (v.0.6.2) software to model time-varying periodic and oscillatory processes by means of a harmonic hidden Markov model (HHMM) as detailed in Hadj-Amar et al. (2020) "Identifying the Recurrence of Sleep Apnea Using a Harmonic Hidden Markov Model " https://arxiv.org/abs/2001.01676

## Contents

HHMM is illustrated in two studies. In the first scenario the data are generated from the model described in Section 2 of the paper and thus this simulation provides a sanity check that the algorithm is indeed retrieveing the correct-prefixed parameters. The second study deals with artificial data from an HMM whose emission distributions are characterized by oscillatory dynamics generate by state-specific autoregressive (AR) time series models. 




* Main Inference Scripts:
```
illustrative_example.jl, ARHMM_example.jl
```
* Functions and Utilities:
```
functions_RJMCMC_SegmentModelSearch.jl, functions_StickyHDPHMM.jl, functions_auxiliary.jl
```


#### Acknowledgments


Parts of this software were adapted from 

1. Emily Fox’s “Sticky HDP-HMM, HDP-AR-HMM and HDP-SLDS”  MATLAB package (https://homes.cs.washington.edu/~ebfox/software/).
2. Beniamino Hadj-Amar's "AutoNOM" Julia package (https://github.com/Beniamino92/AutoNOM).
3. Yee Whye Teh’s "Nonparametric Bayesian Mixture Models" MATLAB package, release 1 (http://www.stats.ox.ac.uk/~teh/software.html).



