##################################################################
# Approach 2 : use xgboost
#
# First, do very minimal preprocessing
# Next, a bit of parameter tuning
# Next, train + predict, N times - retaining the predictions
# Finally, average out the predictions for the final submission
##################################################################

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
# Train the model
train_target <- y
train_eval   <- train[,feature_names]

train_predict <- function(seed_to_use, col_name) {
  set.seed(seed_to_use)

  h <- sample(nrow(train),45600) # validation set to help analyze progress
  dval   <- xgb.DMatrix(data = data.matrix(train_eval[h,]), label = y[h])
  dtrain <- xgb.DMatrix(data = data.matrix(train_eval[-h,]), label=y[-h])

  xgb_watchlist <- list(val=dval, train=dtrain)

  xgb_params <- list(
    objective           = "binary:logistic",
    booster             = "gbtree",
    eval_metric         = "logloss",
    eta                 = config$eta,
    max_depth           = config$max_depth,
    subsample           = config$subsample,
    colsample_bytree    = config$colsample_bytree,
    min_child_weight    = config$min_child_weight
  )

  xgb_model <- xgb.train(
    params              = xgb_params,
    data                = dtrain,
    nrounds             = 1000000,
    verbose             = 1,
    watchlist           = xgb_watchlist,
    print.every.n       = config$print_every_n,
    early.stop.round    = config$early_stop_round,
    maximize            = FALSE
  )

  message(paste("valuation set size", length(h),
                " max_depth", xgb_params$max_depth,
                " eta", xgb_params$eta,
                " subsample", xgb_params$subsample,
                " min_child_weight", xgb_params$min_child_weight,
                " colsample_bytree", xgb_params$colsample_bytree,
                " best score", xgb_model$bestScore,
                " best ind", xgb_model$bestInd,
                " seed", seed_to_use))

  predictions <- predict(xgb_model, data.matrix(test[,feature_names]))
  output_df  <- data.frame(ID=test_id, PredictedProb=predictions)
  names(output_df)[names(output_df)=="PredictedProb"] <- col_name
  list("xgb_model" = xgb_model, "output_df" = output_df)
}

# Run for N iterations
message(paste("Start of",config$num_iterations,"iterations."))
final_df <- NULL
cols <- c()
bestScores <- c()
prfx <- "PredictedProb"
for (i in 1:config$num_iterations) {
  seed_to_use = config$seed + (i*693)
  col_name <- paste0(prfx,i)
  cols <- c(cols, col_name)
  l <- train_predict(seed_to_use, col_name)
  output_df <- l$output_df
  bestScores <- c(bestScores, l$xgb_model$bestScore)
  if (is.null(final_df)) { final_df <- output_df }
  else { final_df <- merge(final_df, output_df) }
}

##################################################################
# Get the average of all predictions
message(paste("Getting average of",config$num_iterations,"iterations."))
final_df$PredictedProb <- rowMeans(subset(final_df, select = cols),
                                   na.rm = TRUE)
final_df <- final_df[,!(names(final_df) %in% cols)]

# Writing the submissions file
current_ts = format(Sys.time(), "%a_%d%b%Y_%H%M")
filename = paste0("submissions/XGB_",current_ts,".csv.gz")
write.csv(final_df, gzfile(filename), quote=FALSE, row.names=FALSE)
message(paste("Finished running script for BNP Approach 2. See",filename))

# Comparing
benchmark <- read_csv(config$benchmark_file)
logloss <- LogLossBinary(as.vector(benchmark[["PredictedProb"]]),
                         as.vector(final_df[["PredictedProb"]]))
message(paste("Log loss against benchmark :", logloss,"\n"))
