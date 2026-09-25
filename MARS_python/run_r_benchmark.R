# R-side benchmark: run MARS (SSNAL) on X.csv, write Omegapath + timing
# usage: Rscript run_r_benchmark.R <x.csv> <out_prefix> [dense|sparse]
suppressMessages(library(MARS))
args <- commandArgs(TRUE)
xin <- args[1]; out <- args[2]
mode <- if (length(args) >= 3) args[3] else "dense"
X <- as.matrix(read.csv(xin, header = FALSE))
p <- nrow(X); n <- ncol(X)
lams <- seq(1, 0.5, -0.1)
sol <- MARS(X, Lambdapath = lams, stopmethod = "fix", fixnumber = length(lams),
            maxiter = 10, printmain = FALSE, maxlambdacheck = TRUE)
lams_out <- sol$Lambdapath
tp <- sol$timepath
write.table(data.frame(lambda = lams_out, time = tp), paste0(out, "_path.csv"),
            row.names = FALSE, col.names = TRUE, sep = ",")
for (k in seq_along(sol$Omegapath)) {
  Om <- as.matrix(sol$Omegapath[[k]])
  if (mode == "sparse") {
    idx <- which(abs(Om) > 1e-8, arr.ind = TRUE)
    tri <- idx[idx[, 1] <= idx[, 2], , drop = FALSE]   # upper triangle incl diag
    df <- data.frame(i = tri[, 1], j = tri[, 2], v = Om[tri])
    write.table(df, paste0(out, "_Om_", k, "_sp.csv"), row.names = FALSE,
                col.names = FALSE, sep = ",")
  } else {
    write.table(Om, paste0(out, "_Om_", k, ".csv"), row.names = FALSE,
                col.names = FALSE, sep = ",")
  }
}
cat("R p=", p, " n=", n, " path length=", length(lams_out), " total time=",
    round(sum(tp), 3), "s\n", sep = "")
