\name{prostmat}
\alias{prostmat}
\docType{data}
\title{Prostate Cancer Gene Expression Data}
\description{
  Gene expression measurements from the prostate cancer study of
  Singh \emph{et al.} (2002). The matrix has \eqn{p = 6033} rows
  (genes) and 102 columns (subjects): the first 50 columns are the
  control subjects and the remaining 52 columns are the prostate
  cancer subjects.
}
\usage{prostmat}
\format{A numeric matrix with 6033 rows and 102 columns.}
\source{
  Singh D, Febbo PG, Ross K, \emph{et al.} (2002). Gene expression
  correlates of clinical prostate cancer behavior. \emph{Cancer Cell},
  1(2):203--209.
}
\examples{
data(prostmat)
X1 <- prostmat[, 1:50]    # control subjects, n = 50
X2 <- prostmat[, 51:102]  # prostate cancer subjects, n = 52
}
\keyword{datasets}
