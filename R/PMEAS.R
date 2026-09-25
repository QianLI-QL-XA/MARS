#' @title Estimation of a sequence of precision matrices (SSNAL, iADMM, eADMM)
#' @description Under the high-dimensional setting, this function is designed
#'   for generating a solution path of the precision matrices by a second-order
#'   algorithm with the l1-penalized D-trace loss.
#' @details Input a p times n sample matrix and a vector of tuning parameters to
#'   obtain a path of estimated precision matrices, where p is the sample
#'   dimension and n is the sample size. Note that our algorithm is designed for
#'   the case that p is much greater than n.
#' @param X A p times n sample matrix.
#' @param stoptol A small value to control the stop tolerance.
#' @param Lambdapath A vector of tuning parameters (positive, sorted in
#'   decreasing order).
#' @param calmethod A string for choosing the computation method from "SSNAL",
#'   "iADMM", or "eADMM".
#' @param maxiter An integer setting the maximum iteration number for the main
#'   loop.
#' @param stopmethod A string to control the stop method, could be "bigs" or
#'   "fix". If it is "bigs", then the algorithm will stop when the number of
#'   non-zero off-diagonal components exceeds n times fixnumber. If it is "fix",
#'   then the algorithm will only calculate the estimator with the first
#'   fixnumber biggest tuning parameters.
#' @param printmain A boolean for controlling whether to print the main results
#'   during the calculation or not.
#' @param printsub A boolean for controlling whether to print the inner problem
#'   results during the calculation or not.
#' @param sigma The initial value of parameter sigma (positive).
#' @param fixnumber A positive integer controlling the stopping rule, as
#'   described in \code{stopmethod}.
#' @param maxlambdacheck A boolean to decide whether to narrow the lambda from
#'   its upper bound path or not.
#' @return A list of solutions: Omegapath, Lambdapath, and timepath, which are
#'   the estimated solutions, the tuning parameters, and the computation time
#'   respectively.
#' @references Qian Li, Binyan Jiang, Defeng Sun. MARS: A second-order
#'   reduction algorithm for high-dimensional sparse precision matrices
#'   estimation. Journal of Machine Learning Research, 24 (2023) 1-44.
#' @examples
#' X <- matrix(rnorm(80 * 15), 80, 15)
#' sol <- PMEAS(X, Lambdapath = seq(1, 0.5, -0.1), calmethod = "SSNAL",
#'              stopmethod = "fix", fixnumber = 3L, maxiter = 5, printmain = FALSE)
#' sol$Omegapath
#' @export
PMEAS <- function(X = NULL, stoptol = 1e-04, Lambdapath = seq(1, 0.5, -0.1), calmethod = "SSNAL", maxiter = 10,
                  stopmethod = "fix", printmain = FALSE, printsub = FALSE, sigma = 1,
                  fixnumber = 10, maxlambdacheck = TRUE) {

    # ---- check the inputs ----
    if (is.null(X)) {
        stop("Error: doesn't find X!")
    }
    if (!is.matrix(X) || !is.numeric(X)) {
        stop("Error: input X must be a numeric matrix!")
    }
    if (anyNA(X)) {
        stop("Error: input X must not contain missing values (NA/NaN)!")
    }
    if (nrow(X) < 2 || ncol(X) < 2) {
        stop("Error: input X must have at least 2 rows (variables) and 2 columns (samples)!")
    }
    if (nrow(X) <= ncol(X)) {
        warning("PMEAS is designed for the high-dimensional setting where the sample dimension p is much greater than the sample size n.")
    }

    if (!is.numeric(Lambdapath) || length(Lambdapath) == 0) {
        stop("Error: Lambdapath must be a non-empty numeric vector!")
    }
    if (!all(Lambdapath > 0) || any(!is.finite(Lambdapath))) {
        stop("Error: Lambdapath must contain only positive finite values!")
    }
    if (is.unsorted(rev(Lambdapath))) {
        warning("Lambdapath is not sorted in decreasing order; the algorithm expects a decreasing sequence for warm starting.")
    }

    if (length(stoptol) != 1 || !is.numeric(stoptol) || !is.finite(stoptol)) {
        stop("Error: stoptol must be a single finite value!")
    }
    if (stoptol <= 0) {
        stop("Error: stoptol must be positive!")
    }
    if (stoptol >= 1) {
        stop("Error: stoptol must be less than 1!")
    }

    if (!calmethod %in% c("SSNAL", "iADMM", "eADMM")) {
        stop("Error: calmethod must be one of {\"SSNAL\", \"iADMM\", \"eADMM\"}!")
    }

    if (length(maxiter) != 1 || !is.numeric(maxiter) || !is.finite(maxiter) ||
        maxiter <= 0 || maxiter != floor(maxiter)) {
        stop("Error: maxiter must be a positive integer!")
    }

    if (!stopmethod %in% c("fix", "bigs")) {
        stop("Error: stopmethod must be one of {\"fix\", \"bigs\"}!")
    }

    if (length(fixnumber) != 1 || !is.numeric(fixnumber) || !is.finite(fixnumber) ||
        fixnumber <= 0 || fixnumber != floor(fixnumber)) {
        stop("Error: fixnumber must be a positive integer!")
    }
    fixnumber <- as.integer(fixnumber)

    if (!is.logical(printmain) || length(printmain) != 1) {
        stop("Error: printmain must be \"TRUE\" or \"FALSE\"!")
    }
    if (!is.logical(printsub) || length(printsub) != 1) {
        stop("Error: printsub must be \"TRUE\" or \"FALSE\"!")
    }

    if (length(sigma) != 1 || !is.numeric(sigma) || !is.finite(sigma) || sigma <= 0) {
        stop("Error: sigma must be a single positive numeric value!")
    }

    if (!is.logical(maxlambdacheck) || length(maxlambdacheck) != 1) {
        stop("Error: maxlambdacheck must be \"TRUE\" or \"FALSE\"!")
    }

    pathsolution <- PMEASc(X, stoptol, Lambdapath, calmethod, stopmethod, maxiter, printmain,
                           printsub, sigma, fixnumber, maxlambdacheck)

    return(pathsolution)
}
