################################## beta regression ###############################
# define custom one inflated likelihood family
library(brms)
library(rstan)
library(ggpubr)
library(bayesplot)
############## define custom likelihood family and stan funtions ################

beta_one_inflated <- custom_family(
    "beta_one_inflated",
    dpars = c('mu', 'phi', 'coi'),
    links = c('logit', 'log', 'logit'),
    lb = c(NA, 0, 0),
    type='real'
)
# with thanks to inspo from https://github.com/paul-buerkner/brms/blob/master/inst/chunks/fun_zero_one_inflated_beta.stan
stan_density <- "
real beta_one_inflated_lpdf(real y, real mu, real phi, real coi) {
    row_vector[2] shape = [mu * phi, (1 - mu) * phi];
    if (y == 1){
        return bernoulli_lpmf(1 | coi);
    }
    else {
        return bernoulli_lpmf(0 |coi) + beta_lpdf(y | shape[1], shape[2]);
    }
    }
real beta_one_inflated_rng(real mu, real phi, real coi) {
    row_vector[2] shape = [mu * phi, (1 - mu) * phi];
    return beta_rng(shape[1], shape[2]);
}
"

stanvars <- stanvar(scode = stan_density, block = "functions")


posterior_predict_beta_one_inflated <- function(i, prep, ...) {
    mu <- brms::get_dpar(prep, "mu", i = i)
    phi <- brms::get_dpar(prep, "phi", i = i)
    coi <- brms::get_dpar(prep, "coi", i = i)
    beta_one_inflated_rng(mu, phi, coi)
}

log_lik_beta_one_inflated <- function(i, prep) {
    mu <- brms::get_dpar(prep, "mu", i = i)
    phi <- brms::get_dpar(prep, "phi", i = i)
    coi <- brms::get_dpar(prep, "coi", i = i)
    y <- prep$data$Y[i]
    beta_one_inflated_lpdf(y, mu, phi, coi)
}