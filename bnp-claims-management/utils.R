##################################################################
# Utility functions
##################################################################

avg_of_N_best_worst_scores <- function(bestScores, final_df, N=3) {
  if (N!=3) {
    N = 3 # TODO handle this limitation
  }
  message(paste("Getting the average of the best", N, "and worst", N, "from",
                length(bestScores), "over",
                config$num_iterations, "iterations."))
  x <- bestScores
  best_N_scores <- c(minN(x,1),minN(x,2),minN(x,3))
  worst_N_scores <- c(maxN(x,1),maxN(x,2),maxN(x,3))
  message(paste(" Best 3 scores :", paste(best_N_scores, collapse = ",")))
  message(paste("Worst 3 scores :", paste(worst_N_scores, collapse = ",")))
  best_N_score_cols <- which(x %in% best_N_scores)
  worst_N_score_cols <- which(x %in% worst_N_scores)
  cols2 <- c(unlist(lapply(best_N_score_cols, function(l) {paste0(prfx,l)})),
             unlist(lapply(worst_N_score_cols, function(l) {paste0(prfx,l)})))
  #message(paste("columns of interest", paste(cols2, collapse = ",")))
  final_df$PredictedProb <- rowMeans(subset(final_df, select = cols2),
                                     na.rm = TRUE)
  list("final_df"=final_df, "best_N_scores"=best_N_scores,
       "best_N_score_cols"=best_N_score_cols)
}


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


maxN <- function(x, N=2) {
  # Credit : http://stackoverflow.com/a/21005136/799188
  len <- length(x)
  if(N>len){
    warning('N greater than length(x).  Setting N=length(x)')
    N <- length(x)
  }
  sort(x,partial=len-N+1)[len-N+1]
}


minN <- function(x, N=2) {
  len <- length(x)
  if(N>len){
    warning('N greater than length(x).  Setting N=length(x)')
    N <- length(x)
  }
  sort(x)[N]
}


print_xgb_params <- function(xgb_params) {
  # Returns a comma-separated string representing the specified xgb_params list
  paste0("max_depth=", xgb_params$max_depth,
        ", eta=", xgb_params$eta,
        ", subsample=", xgb_params$subsample,
        ", min_child_weight=", xgb_params$min_child_weight,
        ", colsample_bytree=", xgb_params$colsample_bytree)
}
