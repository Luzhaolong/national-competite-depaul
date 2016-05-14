install.packages('randomForest')
install.packages('dummies')
install.packages('Boruta')
install.packages('ROCR')
install.packages('glmnet')
install.packages('caret')
install.packages('e1071')
install.packages('corrplot')
install.packages('dplyr')

library(caret)
library(Boruta)
library(rpart)
library(caret)
library(e1071)
library(corrplot)
library(dummies)
library(nnet)
library(randomForest)
library(dplyr)
library(ROCR)
library(glmnet)


###############################################################################################################
################################  MODELING  USING ALL GOOD BORUTA FEATURES#####################################
###############################################################################################################
traindata <- read.csv('train_Boruta001150.csv',sep=',',header=TRUE,row.names = NULL)[,-1]
#View(traindata)
validata <- read.csv('validation_Boruta001150.csv',sep=',',header=TRUE,row.names = NULL)[,-1]
#View(validata)
traindatafull <- read.csv('train_Borutaall001150.csv',sep=',',header=TRUE)[,-1]
validatafull <- read.csv('validation_Borutaall001150.csv',sep=',',header=TRUE)[,-1]

traindata1<- read.csv('train_Boruta005120.csv',sep=',',header=TRUE,row.names = NULL)[,-1]
#View(traindata)
validata1 <- read.csv('validation_Boruta005120.csv',sep=',',header=TRUE,row.names = NULL)[,-1]
#View(validata)
traindatafull1 <- read.csv('train_Borutaall005120.csv',sep=',',header=TRUE)[,-1]
validatafull1 <- read.csv('validation_Borutaall005120.csv',sep=',',header=TRUE)[,-1]


#FIXED Y
traindata$Active_Customer <- as.factor(traindata$Active_Customer)
validata$Active_Customer <- as.factor(validata$Active_Customer)

#################################################################
################  BENCHEN MRK RANDOMFOREST  #####################
#################################################################

#RandomForest trees
#With ntree = 100 we have Accuracy of 0.6669 with 95%CI[0.6558,0.6779]
RFfit <-randomForest(Active_Customer~.,data = traindata,ntree=100,importance=TRUE, proximity=TRUE)
RFpred <- predict(RFfit,validata)
resRF <- table(observed = validata$Active_Customer,predicted=RFpred)
confusionMatrix(resRF)
#Accuracy:0.6681  Sensitivity:0.6433  Specificity 0.6994

#################################################################
###################  LOGISTIC REGRESSION ########################
#################################################################
#LG to see the features p-value
logrgfit <- glm(Active_Customer~.,family=binomial(link='logit'),data=traindata)
logpred <- predict(logrgfit,validata,type='response')
logpredreslt <- ifelse(logpred >0.5,1,0)
micClassificationerror <- mean(logpredreslt != validata$Active_Customer)
print(paste('Accuracy',1-micClassificationerror))
summary(logrgfit)
#Accuracy:0.6658

#Using Lasso to regularization the regression and finding the gradient descent parameter
X_penlregre = as.matrix(traindata[,1:83])
Y_penlregre = traindata$Active_Customer
X_penlregre_validata = as.matrix(validata[,1:83])
Y_penlregre_validata = validata$Active_Customer

lasso_logrgfit <- glmnet(X_penlregre,Y_penlregre,family='binomial',alpha=1)
#summary(lasso_logrgfit)
lasso_log <- predict(lasso_logrgfit,X_penlregre_validata,type='response')
lasso_logpredreslt <- ifelse(lasso_log >0.5,1,0)
micClassificationerror_lasso <- mean(lasso_logpredreslt != validata$Active_Customer)
print(paste('Accuracy',1-micClassificationerror_lasso))
summary(lasso_logrgfit)
plot.glmnet(lasso_logrgfit)
#Accuracy: 0.6600

ridge_logrgfit <- glmnet(X_penlregre,Y_penlregre,family='binomial',alpha=0)
ridge_log <- predict(ridge_logrgfit,X_penlregre_validata,type='response')
ridge_logpredreslt <- ifelse(ridge_log >0.5,1,0)
micClassificationerror_ridge <- mean(ridge_logpredreslt != validata$Active_Customer)
print(paste('Accuracy',1-micClassificationerror_ridge))
summary(lasso_logrgfit)
plot.glmnet(ridge_logrgfit)
#Accuracy:0.6301


backwards <- step(logrgfit)
summary(backwards)
backwards$anova
RFfitbackward <-randomForest(formula(backwards),data = traindata,ntree=100,importance=TRUE, proximity=TRUE)
RFpredbackward <- predict(RFfitbackward,validata)
resRFbackward <- table(observed = validata$Active_Customer,predicted=RFpredbackward)
confusionMatrix(resRFbackward)
