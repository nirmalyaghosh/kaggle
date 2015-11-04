library(readr)
library(plyr)

set.seed(720)
rm(list=ls())

cat("Reading the train and test data\n")
train  <- read_csv("data/train.csv")
test  <- read_csv("data/test.csv")
cat("Training data dimensions:", dim(train))
cat(" Testing data dimensions:", dim(test))

################################################################################
# Preprocessing starts here

# Add an ID column
ID <- rownames(train)
train <- cbind(ID=ID, train)
train$ID <- as.integer(train$ID)

train$Weekday <- as.factor(train$Weekday)
# weekdays <- c("Monday"=1, "Tuesday"=2, "Wednesday"=3, "Thursday"=4, "Friday"=5, 
#               "Saturday"=6, "Sunday"=7)
# x <- factor(train$Weekday)
# train$Weekday <- revalue(x, weekdays)
train$DepartmentDescription <- as.factor(train$DepartmentDescription)


# nn<-reshape(train,timevar="DepartmentDescription",idvar="ID",direction="wide")
# names(nn)[-1]<-as.character(train$DepartmentDescription)
# nn[is.na(nn)]<-0

dept_desc <- train$DepartmentDescription
mat <- matrix(0, nrow=length(dept_desc), ncol=length(unique(dept_desc)))
colnames(mat) <- as.character(unique(dept_desc))
y <- tapply(dept_desc, INDEX=dept_desc, FUN=function(x){
  temp <- which(dept_desc==x[1])
  mat[temp, as.character(x[1])] <<- 1
})
matdf <- as.data.frame(mat)
matdf <- cbind(ID=ID, matdf)
matdf$ID <- as.integer(matdf$ID)
train <- merge(train, matdf)


################################################################################
# Sampling training data - to create training and validation datasets
cat("sampling train to get atraining and validation datasets\n")
smp_size <- floor(0.60 * nrow(train))
idx <- sample(seq_len(nrow(train)), size = smp_size)
train2 <- train[idx, ]
val <- train[-idx, ]
rownames(train2) <- NULL
rownames(val) <- NULL
rm(idx, smp_size, ID)

gc()

