# Smoke test for PMEAS: SSNAL, iADMM and eADMM
library(MARS)

set.seed(42)
p <- 40
n <- 10
X <- matrix(rnorm(p * n), p, n)

for (m in c("SSNAL", "iADMM", "eADMM")) {
  sol <- PMEAS(X, Lambdapath = seq(1, 0.4, -0.3), calmethod = m,
               stopmethod = "fix", fixnumber = 3L, maxiter = 3,
               printmain = FALSE, maxlambdacheck = FALSE)
  stopifnot(is.list(sol), length(sol) == 3)
  stopifnot(length(sol$Omegapath) == 3, length(sol$Lambdapath) == 3)
  for (k in seq_along(sol$Omegapath)) {
    Om <- as.matrix(sol$Omegapath[[k]])
    stopifnot(nrow(Om) == p, ncol(Om) == p)
    stopifnot(isTRUE(all.equal(Om, t(Om), tolerance = 1e-10)))   # symmetry
    stopifnot(all(is.finite(Om)))                                  # finite entries
  }
}
cat("PMEAS smoke test passed.\n")
