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
train$Date <- NULL
# Repeat the same for the test dataset
test$Day <- as.integer(format(test$Date, "%d"))
test$Month <- as.integer(format(test$Date, "%m"))
test$Year <- as.integer(format(test$Date, "%Y"))
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

# Change columns which contain characters into integers
feature.names <- names(train)[c(1,2,5:18)]
# NOTE :
# - feature.names has no Sales or Customers columns,
# feature.names <- names(train)[c(1,2,5:19)] #NOTE : No Sales or Customers columns
# message("Replacing categorical text variables with numeric ids")
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
