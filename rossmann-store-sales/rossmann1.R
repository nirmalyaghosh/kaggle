##################################################################
# Rossmann Model 1
# - only using stores that were open (in train.csv.zip)
#
# References:
# https://www.kaggle.com/abhilashawasthi/rossmann-store-sales/xgb-rossmann/run/86608/code
# https://www.kaggle.com/khozzy/rossmann-store-sales/xgboost-parameter-tuning-template/run/90168/notebook
##################################################################

library(readr)
library(rlogging)
library(xgboost)
library(yaml)
library(zoo)

set.seed(720)
rm(list=ls())
SetLogFile("rossmann-log.txt")
message("Reading the train and test data for Rossmann Model 1")
train  <- read_csv("data/train.csv.zip", col_types="nnDnnnncc")
test  <- read_csv("data/test.csv.zip", col_types="nnnDnncc")
store <- read_csv("data/store.csv.zip")
config = yaml.load_file("rossmann.yml")

# Replacing missing values with 0
train[is.na(train)] <- 0
test[is.na(test)]   <- 0

# Add Id column to training data
id <- rownames(train)
train <- cbind(Id=id, train)
train$Id <- as.integer(train$Id)

# Convert the date column into integers
message("Convert the date column into integers - and reorder the columns")
train$Day <- as.integer(format(train$Date, "%d"))
train$Month <- as.integer(format(train$Date, "%m"))
train$Year <- as.integer(format(train$Date, "%Y"))
train <- train[c(1:4,11:13,5:10)]
train$Date <- NULL
# Repeat the same for the test dataset
test$Day <- as.integer(format(test$Date, "%d"))
test$Month <- as.integer(format(test$Date, "%m"))
test$Year <- as.integer(format(test$Date, "%Y"))
test <- test[c(1:4,9:11,5:8)]
test$Date <- NULL

# Convert the StateHoliday column into a Is_StateHoliday column
message("Converting StateHoliday into Is_StateHoliday")
state_holidays <- c("a", "b", "c")
train <- within(train, {
  Is_StateHoliday = ifelse(StateHoliday %in% state_holidays, 1, 0)
})
test <- within(test, {
  Is_StateHoliday = ifelse(StateHoliday %in% state_holidays, 1, 0)
})
train$StateHoliday <- NULL
test$StateHoliday <- NULL
rm(state_holidays)

# Convert CompetitionOpenSince[Month/Year] columns into a single column
# Number of years since competion opened, 'NumYearsSinceCompetitionOpened'
store$CompetitionOpenSince <- as.yearmon(paste(store$CompetitionOpenSinceYear,
                                               store$CompetitionOpenSinceMonth,
                                               sep = "-"))
oct2015 <- as.yearmon("2015-10")
store$NumYearsSinceCompetitionOpened <- oct2015 - store$CompetitionOpenSince
store[,ncol(store)][is.na(store[ncol(store)])] <- -1
store$CompetitionOpenSinceMonth <- NULL
store$CompetitionOpenSinceYear <- NULL
store$CompetitionOpenSince <- NULL

# Convert Promo2Since[Month/Year] columns into a single column
# Number of days since competion opened, 'NumDaysSincePromo2'
message("Converting Promo2Since[Month/Year] into DaysSincePromo2")
store$Promo2Since <- as.POSIXct(paste(store$Promo2SinceYear,
                                      store$Promo2SinceWeek, 1, sep = "-"),
                                      format = "%Y-%U-%u")
store$DaysSincePromo2 <- as.numeric(as.POSIXct("2015-10-01",
                                               format = "%Y-%m-%d") -
                                      store$Promo2Since)
store[,ncol(store)][is.na(store[ncol(store)])] <- -1
store$Promo2SinceWeek <- NULL
store$Promo2SinceYear <- NULL
store$Promo2Since <- NULL

# Preprocessing the store dataset. This involves :
# - assuming average distance when CompetitionDistance is NA
message("Assuming average CompetitionDistance when NA")
store[is.na(store$CompetitionDistance), "CompetitionDistance"] =
  as.integer(mean(na.omit(store$CompetitionDistance)))

# Merge with store dataset
message("Merge with store dataset")
train <- merge(train,store)
test <- merge(test,store)

# Move the Is_StateHoliday column to the end
train <- train[c(1:11,13:19,12)]
test <- test[c(1:9,11:17,10)]

# Only use data where the stores were open and remove row where sales is 0
message("Only use data where the stores were open")
train <- train[ which(train$Open=='1'),] # Reduction from 1017209 to 844392
train <- train[ which(train$Sales!='0'),] # Reduction from 844392 to 844338

# Change columns which contain characters into integers
for (f in c("SchoolHoliday", "StoreType", "Assortment", "PromoInterval")) {
  train[[f]] <- as.integer(as.factor(train[[f]]))
  test[[f]] <- as.integer(as.factor(test[[f]]))
}

# Creating the training and evaluation datasets
val_r = config$model1$train_eval_split
message(sprintf("Using %.2f%% of raw training data for validation",(val_r*100)))
val <- train[sample(nrow(train), as.integer(nrow(train)*val_r)),]
train60p <- subset(train, ! Id %in% val$Id )
gc()

# Making train and validation matrices
message("Making train and validation matrices")
feature.names <- names(train)[c(1,3:6,9:17,18,19)]
# NOTE :
# - feature.names has no Sales or Customers columns,
dtrain <- xgb.DMatrix(data.matrix(train60p[,feature.names]),
                      label=log(train60p$Sales+1))
dval <- xgb.DMatrix(data.matrix(val[,feature.names]), label=log(val$Sales+1))
watchlist <- list(eval = dval, train = dtrain)
gc()

# Parameters for xgboost
param <- list(
  objective        = "reg:linear",
  booster          = "gbtree",
  eta              = config$model1$eta,
  max_depth        = config$model1$max_depth,
  subsample        = config$model1$subsample,
  colsample_bytree = config$model1$colsample_bytree
)

# Evaluation function
RMPSE <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  elab <- exp(as.numeric(labels)) - 1
  epreds <- exp(as.numeric(preds)) - 1
  err <- sqrt(mean((epreds/elab-1)^2))
  return(list(metric = "RMPSE", value = err))
}

# Train the model
message(paste0("Start training model. Params : nrounds=",config$model1$nrounds,
               ", early.stop.round=",config$model1$early_stop_round,
               ", eta=", param$eta, ", max_depth=",param$max_depth,
               ", subsample=",param$subsample,
               ", colsample_bytree=",param$colsample_bytree))
clf <- xgb.train(
  params           = param,
  data             = dtrain,
  nrounds          = config$model1$nrounds,
  early.stop.round = config$model1$early_stop_round,
  verbose          = 1,
  watchlist        = watchlist,
  maximize         = FALSE,
  feval            = RMPSE
)
the_bestScore <- clf$bestScore
the_bestInd <- clf$bestInd
message(paste0("Finished training model. Best score ", the_bestScore,
            ", best iteration ",the_bestInd))

# Making predictions
message("Making predictions")
feature.names <- feature.names[feature.names != "Customers"]
pred_sales <- exp(predict(clf, data.matrix(test[,feature.names]),
                          ntreelimit=the_bestInd)) -1
submission <- data.frame(Id=test$Id, Sales=pred_sales)

# Creating the submissions file
current_ts = format(Sys.time(), "%d%b%Y_%H%M")
filename = paste("submissions/model_1_",current_ts,".csv",sep="")
write.table(submission[order(submission$Id),], filename, row.names=FALSE,
            sep=",")
message(paste("Finished running script for Rossmann Model 1. See",filename,"\n"))
