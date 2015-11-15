##################################################################
# Rossmann Model 2
# - multiple LM models, one per store
# - only using stores that were open (in train.csv.zip)
#
# Based on https://www.kaggle.com/alexxanderlarko/rossmann-store-sales/sample2910/run/96505/code
##################################################################

library(readr)
library(rlogging)

set.seed(720)
rm(list=ls())
SetLogFile("rossmann-log.txt")
message("Reading the train and test data for Rossmann Model 2")
train  <- read_csv("data/train.csv.zip", col_types="nnDnnnncc")
test  <- read_csv("data/test.csv.zip", col_types="nnnDnncc")
store <- read_csv("data/store.csv.zip")

# Merge with store dataset
train <- merge(train,store)
test <- merge(test,store)

# Replacing missing values with 0
train[is.na(train)] <- 0
test[is.na(test)]   <- 0

# Only use data where the stores were open and remove row where sales is 0
message("Only use data where the stores were open")
train <- train[ which(train$Open=='1'),]
train <- train[ which(train$Sales!='0'),]

# Convert the date column into integers
message("Convert the date column into integers - and reorder the columns")
train$Day <- as.integer(format(train$Date, "%d"))
train$Month <- as.integer(format(train$Date, "%m"))
train$Year <- as.integer(format(train$Date, "%Y"))
# Repeat the same for the test dataset
test$Day <- as.integer(format(test$Date, "%d"))
test$Month <- as.integer(format(test$Date, "%m"))
test$Year <- as.integer(format(test$Date, "%Y"))

# Removing the Date and StateHoliday columns (since elements are extracted)
train$Date <- NULL
train$StateHoliday <- NULL
test$Date <- NULL
test$StateHoliday <- NULL

# Change columns which contain characters into integers
feature.names <- names(train)[c(1,2,5:19)] #NOTE : No Sales or Customers columns
message("Replacing categorical text variables with numeric ids")
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    train[[f]] <- as.integer(as.factor(train[[f]]))
    test[[f]] <- as.integer(as.factor(test[[f]]))
  }
}

# Train the models and making predictions - one per store
message(paste("Start training", length(unique(test$Store)), "models"))
buf <- NULL
for(store_id in unique(test$Store)) {
  set.seed(200)
  store_train <- train[train$Store==store_id,]
  store_test <- test[test$Store==store_id,]
  label <- log(store_train$Sales+1)
  clf <- lm(label~., data=store_train[,feature.names])
  store_pred <- exp(predict(clf, store_test[,feature.names])) -1
  store_test$Sales <- store_pred
  buf <- rbind(buf, store_test)
}
submission <- data.frame(Id=buf$Id, Sales=buf$Sales)
message("Finished training models")

# Creating the submissions file
current_ts = format(Sys.time(), "%d%b%Y_%H%M")
filename = paste("submissions/model_2_",current_ts,".csv",sep="")
submission <- submission[order(submission$Id),]
write_csv(submission, filename)
message(paste("Finished running script for Rossmann Model 2. See",filename,"\n"))
