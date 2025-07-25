% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/stats.R
\name{incidence}
\alias{incidence}
\title{Compute the incidence matrix of a Phylo2Vec vector.}
\usage{
incidence(vector, format = "C")
}
\arguments{
\item{format}{The format of the incidence matrix.
Options are "coo"/"T" (coordinate), "csr"/"R" (compressed sparse row),
"csc"/"C" (compressed sparse column), "dense"/"D" (dense matrix).
Default is "C" for compressed sparse row.}

\item{vector_or_matrix}{Phylo2Vec vector (ndim == 1)}
}
\value{
Incidence matrix in the specified format.
}
\description{
Compute the incidence matrix of a Phylo2Vec vector.
}
