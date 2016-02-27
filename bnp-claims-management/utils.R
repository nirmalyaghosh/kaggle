##################################################################
# Utility functions
##################################################################

az_to_int <- function(az) {
  # Convert from Hexavigesimal (base 26) to base 10
  # Credit : https://www.kaggle.com/c/bnp-paribas-cardif-claims-management/
  #          forums/t/18734/looks-like-a-nice-challenge/108917#post108917
  xx <- strsplit(tolower(az), "")[[1]]
  pos <- match(xx, letters[(1:26)])
  result <- sum( pos* 26^rev(seq_along(xx)-1))
  return(result)
}


LogLossBinary = function(actual, predicted, eps = 1e-15) {
  # Credit : http://www.r-bloggers.com/making-sense-of-logarithmic-loss/
  predicted = pmin(pmax(predicted, eps), 1-eps)
  - (sum(actual * log(predicted) + (1 - actual) * log(1 - predicted))) / length(actual)
}
