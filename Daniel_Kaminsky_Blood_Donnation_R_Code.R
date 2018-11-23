################################################
########       Daniel Kaminsky         #########
########  DrivenData Name: NW_DanielK ##########
################################################

# Load packages
library(C50)
library(car)
library(caret)
library(dplyr)
library(foreach)
library(gbm)
library(glmnet)
library(kernlab)
library(MASS)
library(Matrix)
library(Metrics)
library(plyr)
library(randomForest)
library(readr)
library(rpart)

################################################
###### Exploratory Data Analysis (EDA) #########
################################################

# Load data
raw.data <- read.csv("D:/TrainingData.csv", sep = ",")
str(raw.data) # 'data.frame':	576 obs. of  6 variables
head(raw.data)
summary(raw.data)

# Missing Values - Count per Variable
sapply(raw.data, function(raw.data) sum(is.na(raw.data)))

# Histogram, Q-Q and Box Plots of Recency
par(mfrow = c(2, 2), mar = c(5.1, 6.1, 4.1, 2.1))
hist(raw.data$Months.since.Last.Donation, col = "Gold", main = "Histogram - Recency", xlab = "Recency")
qqnorm(raw.data$Months.since.Last.Donation, col = "darkblue", pch = 'o', main = "Q-Q Plot - Recency")
qqline(raw.data$Months.since.Last.Donation, col = "darkred", lty = 2, lwd = 3)
boxplot(raw.data$Months.since.Last.Donation, col = "red", pch = 16,
        main = "Box Plot - Recency")
par(mfrow = c(1, 1), mar = c(5.1, 4.1, 4.1, 2.1))

# Histogram, Q-Q and Box Plots of Frequency
par(mfrow = c(2, 2), mar = c(5.1, 6.1, 4.1, 2.1))
hist(raw.data$Number.of.Donations, col = "Gold", main = "Histogram - Frequency", xlab = "Frequency")
qqnorm(raw.data$Number.of.Donations, col = "darkblue", pch = 'o', main = "Q-Q Plot - Frequency")
qqline(raw.data$Number.of.Donations, col = "darkred", lty = 2, lwd = 3)
boxplot(raw.data$Number.of.Donations, col = "red", pch = 16,
        main = "Box Plot - Frequency")
par(mfrow = c(1, 1), mar = c(5.1, 4.1, 4.1, 2.1))

# Histogram, Q-Q and Box Plots of Monetary
par(mfrow = c(2, 2), mar = c(5.1, 6.1, 4.1, 2.1))
hist(raw.data$Total.Volume.Donated..c.c.., col = "Gold", main = "Histogram - Monetary", xlab = "Monetary")
qqnorm(raw.data$Total.Volume.Donated..c.c.., col = "darkblue", pch = 'o', main = "Q-Q Plot - Monetary")
qqline(raw.data$Total.Volume.Donated..c.c.., col = "darkred", lty = 2, lwd = 3)
boxplot(raw.data$Total.Volume.Donated..c.c.., col = "red", pch = 16,
        main = "Box Plot - Monetary")
par(mfrow = c(1, 1), mar = c(5.1, 4.1, 4.1, 2.1))

# Histogram, Q-Q and Box Plots of Time
par(mfrow = c(2, 2), mar = c(5.1, 6.1, 4.1, 2.1))
hist(raw.data$Months.since.First.Donation, col = "Gold", main = "Histogram - Time", xlab = "Time")
qqnorm(raw.data$Months.since.First.Donation, col = "darkblue", pch = 'o', main = "Q-Q Plot - Time")
qqline(raw.data$Months.since.First.Donation, col = "darkred", lty = 2, lwd = 3)
boxplot(raw.data$Months.since.First.Donation, col = "red", pch = 16,
        main = "Box Plot - Time")
par(mfrow = c(1, 1), mar = c(5.1, 4.1, 4.1, 2.1))

# Histograms of Recency, Frequency, Monetary, and Time
par(mfrow = c(2, 2), mar = c(5.1, 6.1, 4.1, 2.1))
hist(raw.data$Months.since.Last.Donation, col = "Gold", main = "Histogram - Recency", xlab = "Recency")
hist(raw.data$Number.of.Donations, col = "steelblue1", main = "Histogram - Frequency", xlab = "Frequency")
hist(raw.data$Total.Volume.Donated..c.c.., col = "palegreen", main = "Histogram - Monetary", xlab = "Monetary")
hist(raw.data$Months.since.First.Donation, col = "firebrick1", main = "Histogram - Time", xlab = "Time")
par(mfrow = c(1, 1), mar = c(5.1, 4.1, 4.1, 2.1))

# Simplifying a couple of Variable Names
names(raw.data) <- make.names(names(raw.data))
colnames(raw.data)[1] <- "id"
colnames(raw.data)[6] <- "Class"
str(raw.data) # 'data.frame':	576 obs. of  6 variables
head(raw.data)

# Split data into train set and validation set (80:20 split)
set.seed(7)
validationIndex <- createDataPartition(raw.data$Class, p=0.80, list=FALSE)
validation.set <- raw.data[-validationIndex,]
train.set <- raw.data[validationIndex,]

# EDA train.set
str(train.set) # 'data.frame':	461 obs. of  6 variables
head(train.set)
summary(train.set)

# Histograms of Recency, Frequency, Monetary, and Time (train.set)
par(mfrow = c(2, 2), mar = c(5.1, 6.1, 4.1, 2.1))
hist(train.set$Months.since.Last.Donation, col = "Gold", main = "Histogram - Recency", xlab = "Recency")
hist(train.set$Number.of.Donations, col = "steelblue1", main = "Histogram - Frequency", xlab = "Frequency")
hist(train.set$Total.Volume.Donated..c.c.., col = "palegreen", main = "Histogram - Monetary", xlab = "Monetary")
hist(train.set$Months.since.First.Donation, col = "firebrick1", main = "Histogram - Time", xlab = "Time")
par(mfrow = c(1, 1), mar = c(5.1, 4.1, 4.1, 2.1))

# Scatterplot Matrix
jittered_x <- sapply(train.set[,2:5], jitter)
pairs(jittered_x, names(train.set[,2:5]), col=(train.set$Class)+1)

################################################
#### Data Cleansing and Feature Engineering ####
################################################

# Checking for highly correlated variables
cor(train.set[,2:5])
# Removing the correlated column
train.set$Total.Volume.Donated..c.c.. <- NULL
head(train.set)

# New Variable: donations.per.month
train.set <- train.set %>% mutate(donations.per.month = Number.of.Donations/Months.since.First.Donation)
# New Variable: tenure.ratio (The ratio of last to first donation months)
train.set <- train.set %>% mutate(tenure.ratio = Months.since.Last.Donation/Months.since.First.Donation)
head(train.set)

# Recode the class labels to Yes/No (required when using classProbs=TRUE in trainControl)
required.labels <- train.set['Class']
recoded.labels <- car::recode(required.labels$Class, "0='No'; 1 = 'Yes'")
train.set$Class <- recoded.labels
train.set$Class  <-as.factor(train.set$Class) # Make the class variable a factor
str(train.set)
head(train.set)

# Removing the id column
train.set$id <- NULL
head(train.set)

################################################
########## Modeling the Train Dataset ##########
################################################

# Baseline Models
# 10-fold cross validation with 3 repeats
trainControl <- trainControl(method="repeatedcv", summaryFunction=mnLogLoss,
                             number=10, repeats=3, classProbs=TRUE)
metric <- "logLoss"

# Train using Logistic Regression (glm), Linear Discriminant Analysis (lda),
# Regularized Logistic Regression (glmnet), Classification and Regression Trees (rpart),
# and Support Vector Machines with Radial Basis Functions (svmRadial) using the caret package.
set.seed(7)
fit.glm <- train(Class~., data=train.set, method="glm", 
                 metric=metric, trControl=trainControl) # GLM

set.seed(7)
fit.lda <- train(Class~., data=train.set, method="lda", 
                 metric=metric, trControl=trainControl) # LDA

set.seed(7)
fit.glmnet <- train(Class~., data=train.set, method="glmnet", 
                    metric=metric,trControl=trainControl)  # GLMNET

set.seed(7)
fit.cart <- train(Class~., data=train.set, method="rpart", 
                  metric=metric,trControl=trainControl)  # CART

set.seed(7)
fit.svm <- train(Class~., data=train.set, method="svmRadial", 
                 metric=metric, trControl=trainControl)  # SVM

# Compare algorithms
results <- resamples(list(LG=fit.glm, LDA=fit.lda, 
                          GLMNET=fit.glmnet, CART=fit.cart, SVM=fit.svm))
summary(results)
dotplot(results, main="logLoss Chart")

# Using Box Cox Transformation to correct the Skewness in the data to try to improve the results
preProcess="BoxCox" 
# 10-fold cross validation with 3 repeats
trainControl <- trainControl(method="repeatedcv", summaryFunction=mnLogLoss,
                             number=10, repeats=3, classProbs=TRUE)
metric <- "logLoss"

set.seed(7)
fit.glm <- train(Class~., data=train.set, method="glm", 
                 metric=metric, trControl=trainControl, preProc=preProcess) # GLM

set.seed(7)
fit.lda <- train(Class~., data=train.set, method="lda", 
                 metric=metric, trControl=trainControl, preProc=preProcess) # LDA

set.seed(7)
fit.glmnet <- train(Class~., data=train.set, method="glmnet", 
                    metric=metric,trControl=trainControl, preProc=preProcess)  # GLMNET

set.seed(7)
fit.cart <- train(Class~., data=train.set, method="rpart", 
                  metric=metric,trControl=trainControl, preProc=preProcess)  # CART

set.seed(7)
fit.svm <- train(Class~., data=train.set, method="svmRadial", 
                 metric=metric, trControl=trainControl, preProc=preProcess)  # SVM

# Compare algorithms
results <- resamples(list(LG=fit.glm, LDA=fit.lda, 
                          GLMNET=fit.glmnet, CART=fit.cart, SVM=fit.svm))
summary(results)
dotplot(results, main="logLoss Chart")

# GLMNET Model Tuning
preProcess="BoxCox"

# 10-fold cross validation with 3 repeats
trainControl <- trainControl(method="repeatedcv", summaryFunction=mnLogLoss, 
                             number=10, repeats=3, classProbs=TRUE)
metric <- "logLoss"

# Tuning the Regularized Logistic Regression by changing the
# mixing percentage (alpha) and the regularization parameter (lambda)
set.seed(7)
grid <- expand.grid(alpha=c((70:100)/100), lambda = c(0.001, 0.0009, 0.0008,0.0007))
fit.glmnet <- train(Class~., data=train.set, method="glmnet", metric=metric, tuneGrid=grid,
                    preProc=preProcess, trControl=trainControl)
plot(fit.glmnet)

best.aplha <- fit.glmnet$bestTune$alpha # 1
best.lambda <- fit.glmnet$bestTune$lambda # 8e-04

################################################
####### Modeling the Validation Dataset ########
################################################

# Reproducing the data cleansing performed in the train.set
# Removing the correlated column
validation.set$Total.Volume.Donated..c.c.. <- NULL

# Redo the feature engineering
validation.set <- validation.set %>% mutate(donations.per.month = Number.of.Donations/Months.since.First.Donation)
validation.set <- validation.set %>% mutate(tenure.ratio = Months.since.Last.Donation/Months.since.First.Donation)

# Recode the class labels to Yes/No
required.labels <- validation.set['Class']
recoded.labels <- car::recode(required.labels$Class, "0='No'; 1 = 'Yes'")
validation.set$Class <- recoded.labels
validation.set$Class  <-as.factor(validation.set$Class)
set.seed(7)
test.pred <- predict(fit.glmnet, newdata=validation.set[, c(2:4, 6:7)], type="prob")

# Function to calculate logLoss
LogLoss <- function(actual, predicted, eps=0.00001) {
  predicted <- pmin(pmax(predicted, eps), 1-eps)
  -1/length(actual)*(sum(actual*log(predicted)+(1-actual)*log(1-predicted)))
}

# Recode Class back to 0 and 1 to be able to use the logLoss function
required.labels <- validation.set['Class']
recoded.labels <- car::recode(required.labels$Class, "'No'=0; 'Yes'=1")
validation.set$Class <- recoded.labels

# logLoss of the Validation dataset Prediction
LL <- LogLoss(as.numeric(as.character(validation.set$Class)), test.pred$Yes)
LL # 0.3718009

################################################
#### Predicting with the test.data Dataset #####
################################################

# Load Data
test.data <- read.csv("D:/TestData.csv", sep = ",")
# Checking test.data
str(test.data)
# Missing Values - Count per Variable
sapply(test.data, function(test.data) sum(is.na(test.data)))

# Reproducing the data cleansing performed in the train.set
# Simplifying a couple of Variable Names
names(test.data) <- make.names(names(test.data))
# Removing the correlated column
test.data$Total.Volume.Donated..c.c.. <- NULL
# Redo the feature engineering
test.data <- test.data %>% mutate(donations.per.month = Number.of.Donations/Months.since.First.Donation)
test.data <- test.data %>% mutate(tenure.ratio = Months.since.Last.Donation/Months.since.First.Donation)
test.set <- test.data[,-1]

# Make predicitons
set.seed(7)
predictions <- predict(fit.glmnet, newdata=test.set, type  = "prob")
pred.df <- as.data.frame(predictions$Yes)
pred.df$id <- test.data$NA.

# Preparing the data for writing to csv using the format from the submission format file
submission_format <- read.csv("D:/BloodDonationSubmissionFormat.csv", check.names=FALSE)
head(submission_format)
# Creating the Output file with the " " and "Made Donation in March 2007"
pred.df$id <- test.data[1]
OutputTest1 <- data.frame(pred.df$id, pred.df[1])
colnames(OutputTest1) <- colnames(submission_format)
head(OutputTest1)

### Write OutputTest1 to CSV ###
write.csv(OutputTest1, 
          file = "D:/OutputTest1.csv", 
          row.names = FALSE)






