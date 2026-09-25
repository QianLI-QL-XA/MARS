#' MARS: A Second-Order Reduction Algorithm for Sparse Precision Matrices
#'
#' The MARS package estimates a solution path of sparse precision matrices
#' under the l1-penalized D-trace loss by an adaptive sieving reduction
#' strategy combined with a semismooth Newton augmented Lagrangian algorithm.
#' The main entry point is \code{\link{MARS}}.
#'
#' @keywords internal
#' @useDynLib MARS, .registration = TRUE
#' @importFrom Rcpp evalCpp
#' @importClassesFrom Matrix dgCMatrix
"_PACKAGE"
