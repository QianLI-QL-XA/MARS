# Smoke test for MARS: solution path via the adaptive sieving algorithm
# with the SSNAL subproblem solver
library(MARS)

set.seed(123)
p <- 60
n <- 15
X <- matrix(rnorm(p * n), p, n)

sol <- MARS(X, Lambdapath = seq(1, 0.4, -0.2), stopmethod = "fix",
            fixnumber = 3L, maxiter = 5, printmain = FALSE,
            maxlambdacheck = FALSE)

stopifnot(is.list(sol), length(sol) == 3)
stopifnot(is.list(sol$Omegapath), length(sol$Omegapath) == 3)
stopifnot(length(sol$Lambdapath) == 3, length(sol$timepath) == 3)
stopifnot(all(sol$Lambdapath > 0))

for (k in seq_along(sol$Omegapath)) {
  Om <- as.matrix(sol$Omegapath[[k]])
  stopifnot(nrow(Om) == p, ncol(Om) == p)
  stopifnot(isTRUE(all.equal(Om, t(Om), tolerance = 1e-10)))   # symmetry
  stopifnot(all(is.finite(Om)))                                  # finite entries
  stopifnot(all(diag(Om) > 0))                                  # positive diagonal
}

cat("MARS smoke test passed.\n")
