##################################################################
# Approach 2 : use xgboost
#
# First, do very minimal preprocessing
#        - remove Near Zero-Variance Predictors
#        - remove highly correlated variables
# Next, a bit of parameter tuning
# Next, train + predict, N times - retaining the predictions
# Finally, take average of only the best 3 and worst 3 predictions
##################################################################

library(caret)
library(readr)
library(rlogging)
library(xgboost)
library(yaml)

rm(list = ls())
SetLogFile("notes.txt")
config = yaml.load_file("bnp.yml")
config = config$approach2
message(paste("Running script for BNP Approach 2,", config$description))
set.seed(config$seed)

train <- read_csv("data/train.csv")
y = train$target
train <- train[,-2]
test <- read_csv("data/test.csv")
source("utils.R")

##################################################################
# Bit of Preprocessing
train[is.na(train)] <- -1
test[is.na(test)]   <- -1

train_id <- train$ID
test_id  <- test$ID
train$ID <- NULL
test$ID  <- NULL

train$is_train <- rep(TRUE,nrow(train))
test$is_train  <- rep(FALSE,nrow(test))
df_all <- rbind(train, test)

# Convert character variables to integer
feature_names <- names(df_all)
char_cols <- c()
for (f in feature_names) {
  if (class(df_all[[f]])=="character") {
    char_cols <- c(char_cols, f)
    levels <- unique(c(df_all[[f]]))
    df_all[[f]] <- as.integer(factor(df_all[[f]], levels=levels))
  }
}
message(paste("Converted columns :", paste(char_cols, collapse = ",")))

drops = c("is_train")
train <- df_all[df_all$is_train == TRUE,!names(df_all) %in% drops]
test <- df_all[df_all$is_train == FALSE,!names(df_all) %in% drops]
feature_names <- feature_names[feature_names != "is_train"]

##################################################################
# Find and remove Near Zero-Variance Predictors
nzv <- data.frame(nearZeroVar(df_all, saveMetrics= TRUE))
drops <- row.names(nzv[nzv$nzv == TRUE, ]) # v3, v38, v74
train <- train[!names(train) %in% drops]
test <- test[!names(test) %in% drops]
message(paste("Dropped NZV variables :", paste(drops, collapse = ",")))
feature_names <- feature_names[!feature_names %in% drops]

# Find and remove highly correlated variables
high_corr_cutoff <- config$high_corr_cutoff
highlyCorrVars1 <- findCorrelation(cor(train), cutoff = high_corr_cutoff)
highlyCorrVars2 <- findCorrelation(cor(test), cutoff = high_corr_cutoff)
drops <- intersect(highlyCorrVars1, highlyCorrVars2)
drops <- names(train)[drops]
train <- train[!names(train) %in% drops]
test <- test[!names(test) %in% drops]
message(paste("Dropped", length(drops), "highly correlated variables (cutoff",
              high_corr_cutoff, ") :", paste(drops, collapse = ",")))
feature_names <- feature_names[!feature_names %in% drops]


##################################################################
# Train the model
train_target <- y
train_eval   <- train[,feature_names]

train_cv <- function(xgb_params, seed_to_use) {
  set.seed(seed_to_use)
  h <- sample(nrow(train), config$validation_set_size)
  dval   <- xgb.DMatrix(data = data.matrix(train_eval[h,]), label = y[h])
  dtrain <- xgb.DMatrix(data = data.matrix(train_eval[-h,]), label=y[-h])
  xgb_watchlist <- list(val=dval, train=dtrain)
  message(paste("Training xgboost with", config$num_folds ,
                "fold cross-validation. Seed :", seed_to_use,
                "Validatation set size :", config$validation_set_size))

  bst_cv = xgb.cv(
    params              = xgb_params,
    data                = dtrain,
    nfold               = config$num_folds,
    nrounds             = 1000000,
    verbose             = config$verbosity,
    print.every.n       = config$print_every_n,
    early.stop.round    = config$early_stop_round,
    nthread             = config$nthread,
    maximize            = FALSE
  )

  gc()
  best <- min(bst_cv$test.logloss.mean)
  bestIter <- which(bst_cv$test.logloss.mean==best)

  cat("\n",best, bestIter,"\n")
  print(bst_cv[bestIter])
  message(paste("Best test.logloss.mean :", best, "best iteration", bestIter[1]))

  bestIter-1
}


train_predict <- function(xgb_params, seed_to_use, col_name, num_rounds) {
  set.seed(seed_to_use)
  h <- sample(nrow(train), config$validation_set_size)
  dval   <- xgb.DMatrix(data = data.matrix(train_eval[h,]), label = y[h])
  dtrain <- xgb.DMatrix(data = data.matrix(train_eval[-h,]), label=y[-h])
  xgb_watchlist <- list(val=dval, train=dtrain)

  xgb_model <- xgb.train(
    params              = xgb_params,
    data                = dtrain,
    nrounds             = num_rounds,
    verbose             = config$verbosity,
    watchlist           = xgb_watchlist,
    print.every.n       = config$print_every_n,
    early.stop.round    = config$early_stop_round,
    nthread             = config$nthread,
    maximize            = FALSE
  )

  if (config$verbosity > 0) {
    message(paste0("BestScore ", round(xgb_model$bestScore, digits = 5),
                   ", BestInd ", xgb_model$bestInd,
                   ", Params : ", print_xgb_params(xgb_params)))
  }

  predictions <- predict(xgb_model, data.matrix(test[,feature_names]))
  output_df  <- data.frame(ID=test_id, PredictedProb=predictions)
  names(output_df)[names(output_df)=="PredictedProb"] <- col_name
  list("xgb_model" = xgb_model, "output_df" = output_df)
}

helper <- function(xgb_params, seed_to_use, col_name, target_df, bestScores) {
    # Train and predict for a given set of parameters
    l <- train_predict(xgb_params, seed_to_use, col_name, 1000000)
    output_df <- l$output_df
    bestScores <- c(bestScores, l$xgb_model$bestScore)
    if (is.null(target_df)) { target_df <- output_df }
    else { target_df <- merge(target_df, output_df) }
    list("target_df"=target_df, "bestScores"=bestScores)
}

# Prepare the parameters required for training the model(s)
xgb_params_base <- list(
  objective           = "binary:logistic",
  booster             = "gbtree",
  eval_metric         = "logloss",
  eta                 = config$eta,
  max_depth           = config$max_depth,
  subsample           = config$subsample,
  colsample_bytree    = config$colsample_bytree,
  min_child_weight    = config$min_child_weight
)

# Training xgboost with cross-validation
xgb_params <- xgb_params_base
cv <- train_cv(xgb_params, config$seed)

# Run for N iterations
final_df <- NULL
cols <- c()
bestScores <- c()
prfx <- "PredictedProb"
ctr = 1
xgb_params_list <- list()
message(paste("Run for", config$num_iterations, "iterations"))
for (i in 1:config$num_iterations) {
  col_name <- paste0(prfx, ctr)
  xgb_params_list[[ctr]] <- xgb_params
  cols <- c(cols, col_name)
  seed_to_use = config$seed + (ctr*config$seed_increment)
  xgb_params$seed_to_use = seed_to_use
  r <- helper(xgb_params, seed_to_use, col_name, final_df, bestScores)
  final_df <- r$target_df
  bestScores <- r$bestScores
  ctr <- ctr + 1
}
message(paste("Best scores :", paste(paste(bestScores, collapse = ", "))))
message(paste("Avg. best score :", mean(bestScores)))

# Get the average of the N predictions
final_df$PredictedProb <- rowMeans(subset(final_df, select = cols),
                                   na.rm = TRUE)

# Writing the submissions file
current_ts = format(Sys.time(), "%a_%d%b%Y_%H%M")
filename = paste0("submissions/XGB_",current_ts,".csv.gz")
submissions_df <- final_df
submissions_df <-
  submissions_df[names(submissions_df) %in% c("ID", "PredictedProb")]
write.csv(submissions_df, gzfile(filename), quote = FALSE, row.names = FALSE)
message(paste("Finished running script for BNP Approach 2. See",filename))

# Comparing
benchmark <- read_csv(config$benchmark_file)
logloss <- LogLossBinary(as.vector(benchmark[["PredictedProb"]]),
                         as.vector(final_df[["PredictedProb"]]))
message(paste("Log loss against benchmark :", round(logloss, digits = 5),"\n"))
