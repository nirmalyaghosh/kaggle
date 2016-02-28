##################################################################
# Approach 2 : use xgboost
#
# First, do very minimal preprocessing
# Next, a bit of parameter tuning
# Next, train + predict, N times - retaining the predictions
# Finally, take average of only the best 3 and worst 3 predictions
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

train_predict <- function(xgb_params, seed_to_use, col_name) {
  set.seed(seed_to_use)

  h <- sample(nrow(train), config$validation_set_size)
  dval   <- xgb.DMatrix(data = data.matrix(train_eval[h,]), label = y[h])
  dtrain <- xgb.DMatrix(data = data.matrix(train_eval[-h,]), label=y[-h])

  xgb_watchlist <- list(val=dval, train=dtrain)

  xgb_model <- xgb.train(
    params              = xgb_params,
    data                = dtrain,
    nrounds             = 1000000,
    verbose             = config$verbosity,
    watchlist           = xgb_watchlist,
    print.every.n       = config$print_every_n,
    early.stop.round    = config$early_stop_round,
    maximize            = FALSE
  )

  if (config$verbosity > 0) {
    message(paste0("BestScore ", round(xgb_model$bestScore, digits = 5),
                   ", BestInd ", xgb_model$bestInd,
                   ", Seed ", seed_to_use,
                   ", Params : ", print_xgb_params(xgb_params)))
  }

  predictions <- predict(xgb_model, data.matrix(test[,feature_names]))
  output_df  <- data.frame(ID=test_id, PredictedProb=predictions)
  names(output_df)[names(output_df)=="PredictedProb"] <- col_name
  list("xgb_model" = xgb_model, "output_df" = output_df)
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

# Run for N iterations
final_df <- NULL
cols <- c()
bestScores <- c()
prfx <- "PredictedProb"

helper <- function(xgb_params, seed_to_use, col_name, target_df, bestScores) {
    # Train and predict for a given set of parameters
    l <- train_predict(xgb_params, seed_to_use, col_name)
    output_df <- l$output_df
    bestScores <- c(bestScores, l$xgb_model$bestScore)
    if (is.null(target_df)) { target_df <- output_df }
    else { target_df <- merge(target_df, output_df) }
    list("target_df"=target_df, "bestScores"=bestScores)
}

ctr = 1
xgb_params <- xgb_params_base
xgb_params_list <- list()
if (config$do_param_tuning == TRUE) {
  # XGB parameter tuning enabled
  # Get the candidate params
  csbt <- as.numeric(unlist(as.list(strsplit(config$p_colsample_bytree, ","))))
  etas <- as.numeric(unlist(as.list(strsplit(config$p_eta, ","))))
  mcws <- as.numeric(unlist(as.list(strsplit(config$p_min_child_weights, ","))))
  md <- as.numeric(unlist(as.list(strsplit(config$p_max_depths, ","))))
  sss <- as.numeric(unlist(as.list(strsplit(config$p_sub_samples, ","))))
  num_runs = length(csbt) * length(etas) * length(mcws)  * length(md) *
             length(sss)
  message(paste("XGB parameter tuning enabled. Starting", num_runs,"runs"))
  # Start the parameter tuning work
  for (m in md) {
    for (s in sss) {
      for (mcw in mcws) {
        for (c in csbt) {
          for (e in etas) {
            xgb_params$colsample_bytree = c
            xgb_params$eta = e
            xgb_params$max_depth = as.integer(m)
            xgb_params$min_child_weight = mcw
            xgb_params$subsample = s
            xgb_params_list[[ctr]] <- xgb_params
            col_name <- paste0(prfx, ctr)
            cols <- c(cols, col_name)
            seed_to_use = config$seed + (ctr*693)
            r <- helper(xgb_params, seed_to_use, col_name, final_df, bestScores)
            final_df <- r$target_df
            bestScores <- r$bestScores
            cat(paste("Done",ctr,"of",num_runs,"\n"))
            ctr <- ctr + 1
          }
        }
      }
    }
  }
} else {
  for (i in 1:config$num_iterations) {
    col_name <- paste0(prfx, ctr)
    xgb_params_list[[ctr]] <- xgb_params
    cols <- c(cols, col_name)
    seed_to_use = config$seed + (ctr*693)
    r <- helper(xgb_params, seed_to_use, col_name, final_df, bestScores)
    final_df <- r$target_df
    bestScores <- r$bestScores
    ctr <- ctr + 1
  }
}

##################################################################
# Get the average of best N and worst N predictions
r <- avg_of_N_best_worst_scores(bestScores, final_df)
final_df <- r$final_df
# Print the best parameters
message("Best params :")
for (i in seq(1,length(r$best_N_scores))) {
  best_params <- xgb_params_list[[r$best_N_score_cols[i]]]
  message(paste("\tScore:", round(r$best_N_scores[[i]], digits = 5),
                "\tParams:", print_xgb_params(best_params)))
}

# Writing the submissions file
current_ts = format(Sys.time(), "%a_%d%b%Y_%H%M")
filename = paste0("submissions/XGB_",current_ts,".csv.gz")
write.csv(final_df, gzfile(filename), quote=FALSE, row.names=FALSE)
message(paste("Finished running script for BNP Approach 2. See",filename))

# Comparing
benchmark <- read_csv(config$benchmark_file)
logloss <- LogLossBinary(as.vector(benchmark[["PredictedProb"]]),
                         as.vector(final_df[["PredictedProb"]]))
message(paste("Log loss against benchmark :", round(logloss, digits = 5),"\n"))
