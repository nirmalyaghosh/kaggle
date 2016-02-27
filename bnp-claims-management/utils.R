##################################################################
# Utility functions
##################################################################

LogLossBinary = function(actual, predicted, eps = 1e-15) {
  # Credit : http://www.r-bloggers.com/making-sense-of-logarithmic-loss/
  predicted = pmin(pmax(predicted, eps), 1-eps)
  - (sum(actual * log(predicted) + (1 - actual) * log(1 - predicted))) / length(actual)
}
