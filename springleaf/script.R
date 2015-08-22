# This is getting a decent public LB score so I just want to note that I don't
# really know what I am doing.

# Following the parameters used in a script which score 0.77548, (0.00376 more than my 0.77172)
# https://www.kaggle.com/michaelpawlus/springleaf-marketing-response/xgboost-example-0-76178/run/52365

library(readr)
library(xgboost)

set.seed(720)
setwd("C:/nirmalya/learning/projects/kaggle/springleaf")
rm(list=ls())

cat("Reading the train and test data\n")
train  <- read_csv("input/train.csv.zip")
test  <- read_csv("input/test.csv.zip")
cat("Training data dimensions:", dim(train))
cat(" Testing data dimensions:", dim(test))

feature.names <- names(train)[2:ncol(train)-1]

################################################################################
# Preprocessing starts here

y = train$target

########################
# Preprocessing Step 1 : Identify and remove columns having one unique value
# Credit : https://www.kaggle.com/darraghdog/springleaf-marketing-response/explore-springleaf/notebook
col_uniq_val_count = sapply(train, function(x) length(unique(x)))
constant_feature_cols = names(col_uniq_val_count[col_uniq_val_count==1])
cat("Constant feature columns:", constant_feature_cols)
# VAR_0205, VAR_0207, VAR_0213, VAR_0214, VAR_0840, VAR_0847, VAR_1428
cat(" Unique values, VAR_0205:", unique(train$VAR_0205)) # for inspection
cat("Removing the constant feature columns\n")
#train = subset(train, select=-constant_feature_cols) # gave an error
train = subset(train, select=-c(VAR_0205, VAR_0207, VAR_0213, VAR_0214,
                                VAR_0840, VAR_0847, VAR_1428))
test = subset(test, select=-c(VAR_0205, VAR_0207, VAR_0213, VAR_0214,
                              VAR_0840, VAR_0847, VAR_1428))
feature.names <- names(train)[2:ncol(train)-1]
cat("Training data dimensions:", dim(train))

########################
# # Preprocessing Step 2 : combine the state ('VAR_0241') and numeric zipcode
# #                        ('VAR_0237') into a single column and remove them
# train$zipcode = paste(train$VAR_0237, train$VAR_0241, sep="")
# test$zipcode = paste(test$VAR_0237, test$VAR_0241, sep="")
# # Remove the state and numeric zipcode columns
# train = subset(train, select=-c(VAR_0237, VAR_0241))
# test = subset(test, select=-c(VAR_0237, VAR_0241))
# # Make sure 'zipcode' is the 2nd last column, just before 'target'
# train <- train[ c(head(names(train),-2), "zipcode", "target") ]
# # Refresh the feature names
# feature.names <- names(train)[2:ncol(train)-1]

# Preprocessing Step 2 : combine columns 'VAR_0008', 'VAR_0009', 'VAR_0010',
#                        'VAR_0011' into a single column and remove them
# train$cols8to11 = paste(train$VAR_0008, train$VAR_0009, train$VAR_0010,
#                         train$VAR_0011, sep="")
# test$cols8to11 = paste(test$VAR_0008, test$VAR_0009, test$VAR_0010,
#                        test$VAR_0011, sep="")
# train = subset(train, select=-c(VAR_0008, VAR_0009, VAR_0010, VAR_0011))
# test = subset(test, select=-c(VAR_0008, VAR_0009, VAR_0010, VAR_0011))
# # Make sure 'cols8to11' is the 2nd last column, just before 'target'
# train <- train[ c(head(names(train),-2), "cols8to11", "target") ]
# # Refresh the feature names
# feature.names <- names(train)[2:ncol(train)-1]

########################
# Preprocessing Step 3
cat("Assuming text variables are categorical & replacing them with numeric ids\n")
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

########################
# Preprocessing Step 4
cat("Replacing missing values with -1\n")
train[is.na(train)] <- -1
test[is.na(test)]   <- -1


################################################################################
# Sampling training data - to create training and validation datasets
cat("sampling train to get atraining and validation datasets\n")
val <- train[sample(nrow(train), 60000),]
train80k <- subset(train, ! ID %in% val$ID )
gc()

# Making train and validation matrices
cat("Making train and validation matrices\n")
feature.names <- names(train)[2:ncol(train)-1] # Call once again, after all preprocessing steps
dtrain <- xgb.DMatrix(data.matrix(train80k[,feature.names]), label=train80k$target)
gc()
dval <- xgb.DMatrix(data.matrix(val[,feature.names]), label=val$target)

watchlist <- list(eval = dval, train = dtrain)

param <- list(  objective           = "binary:logistic",
                # booster = "gblinear",
                eta                 = 0.01,
                max_depth           = 20,  # changed from default of 6 to 13,15,17,19,8
                subsample           = 0.7,
                colsample_bytree    = 0.7,
                eval_metric         = "auc"
                # alpha = 0.0001,
                # lambda = 1
)

# # Apply 5-fold cross validation
# clf.cv <- xgb.cv(   params              = param,
#                     data                = dtrain,
#                     label               = y,
#                     nfold               = 7,
#                     nrounds             = 900,
#                     prediction          = TRUE,
#                     verbose             = TRUE)
# tail(clf.cv$dt, n = 15) # Check out the last 15
#
# # Find index of maximum test AUC
# max.test.auc.idx = which.max(clf.cv$dt[, test.auc.mean])
# max.test.auc.idx
# clf.cv$dt[max.test.auc.idx,]

# Train the model
clf <- xgb.train(   params              = param,
                    data                = dtrain,
                    #nrounds             = max.test.auc.idx, # Used to be 1200
                    nrounds             = 2400,
                    verbose             = 1,
                    early.stop.round    = 20,
                    watchlist           = watchlist,
                    maximize            = TRUE)

current_ts = format(Sys.time(), "%a_%d%b%Y_%H%M%S")
xgb.save(clf, paste("xgboost_",current_ts,".model",sep=""))
the_bestScore <- clf$bestScore
the_bestInd <- clf$bestInd

################################################################################
# Making predictions
submission <- data.frame(ID=test$ID)
submission$target <- NA
for (rows in split(1:nrow(test), ceiling((1:nrow(test))/10000))) {
  submission[rows, "target"] <- predict(clf, data.matrix(test[rows,feature.names]), ntreelimit=the_bestInd)
}

# Creating the file for submission
cat("saving the submission file\n")
filename = paste("submissions/",current_ts,".csv",sep="")
write_csv(submission, filename)
