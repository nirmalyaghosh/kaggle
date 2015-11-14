##################################################################
# Rossmann Model 1
# - only using stores that were open (in train.csv.zip)
# - not using columns :
#     CompetitionOpenSince[Month/Year]
#     Promo2Since[Year/Week]
#
# References:
# https://www.kaggle.com/abhilashawasthi/rossmann-store-sales/xgb-rossmann/run/86608/code
# https://www.kaggle.com/khozzy/rossmann-store-sales/xgboost-parameter-tuning-template/run/90168/notebook
##################################################################

library(readr)
library(rlogging)
library(xgboost)

set.seed(720)
rm(list=ls())
SetLogFile("rossmann-log.txt")
message("Reading the train and test data for Rossmann Model 1")
train  <- read_csv("data/train.csv.zip", col_types="nnDnnnncc")
test  <- read_csv("data/test.csv.zip", col_types="nnnDnncc")
store <- read_csv("data/store.csv.zip")

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

# Preprocessing the store dataset. This involves :
# - dropping columns which I can't handle at the moment
# - assuming average distance when CompetitionDistance is NA
message("Dropping columns : CompetitionOpenSince[Month/Year]")
store$CompetitionOpenSinceMonth <- NULL
store$CompetitionOpenSinceYear <- NULL
message("Dropping columns : Promo2Since[Month/Year]")
store$Promo2SinceWeek <- NULL
store$Promo2SinceYear <- NULL
message("Assuming average CompetitionDistance when NA")
store[is.na(store$CompetitionDistance), "CompetitionDistance"] =
  as.integer(mean(na.omit(store$CompetitionDistance)))

# Merge with store dataset
message("Merge with store dataset")
train <- merge(train,store)
test <- merge(test,store)

# Only use data where the stores were open and remove row where sales is 0
message("Only use data where the stores were open")
train <- train[ which(train$Open=='1'),] # Reduction from 1017209 to 844392
train <- train[ which(train$Sales!='0'),] # Reduction from 844392 to 844338

# Change columns which contain characters into integers
for (f in c("StateHoliday", "SchoolHoliday", "StoreType", "Assortment",
            "PromoInterval")) {
  message("Changing column",f," from character to integer type")
  train[[f]] <- as.integer(as.factor(train[[f]]))
  test[[f]] <- as.integer(as.factor(test[[f]]))
}

# Use 40% of the original training data for validation
val <- train[sample(nrow(train), as.integer(nrow(train)*0.4)),]
train60p <- subset(train, ! Id %in% val$Id )
gc()

# Making train and validation matrices
message("Making train and validation matrices")
feature.names <- names(train)[c(1,3:6,8:17)]
dtrain <- xgb.DMatrix(data.matrix(train60p[,feature.names]),
                      label=log(train60p$Sales+1))
dval <- xgb.DMatrix(data.matrix(val[,feature.names]), label=log(val$Sales+1))
watchlist <- list(eval = dval, train = dtrain)
gc()

# Parameters for xgboost
param <- list(
  objective        = "reg:linear",
  booster          = "gbtree",
  eta              = 0.01,
  max_depth        = 8,
  subsample        = 0.7,
  colsample_bytree = 0.7
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
nr = 900
message(paste0("Start training model. Params : nrounds=",nr,
               ", eta=", param$eta, ", max_depth=",param$max_depth,
               ", subsample=",param$subsample,
               ", colsample_bytree=",param$colsample_bytree))
clf <- xgb.train(
  params           = param,
  data             = dtrain,
  nrounds          = nr,
  early.stop.round = 20,
  verbose          = 1,
  watchlist        = watchlist,
  maximize         = FALSE,
  feval            = RMPSE
)
the_bestScore <- clf$bestScore
the_bestInd <- clf$bestInd
message(paste0("Finished training model. Best score ", the_bestScore,
            ", best index ",the_bestInd))

# Making predictions
message("Making predictions")
feature.names <- feature.names[feature.names != "Customers"]
pred_sales <- exp(predict(clf, data.matrix(test[,feature.names]),
                          ntreelimit=the_bestInd)) -1
submission <- data.frame(Id=test$Id, Sales=pred_sales)

# Creating the submissions file
current_ts = format(Sys.time(), "%a_%d%b%Y_%H%M%S")
filename = paste("submissions/",current_ts,".csv",sep="")
write_csv(submission, filename)
message(paste("Finished running script for Rossmann Model 1. See",filename,"\n"))
