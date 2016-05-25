install.packages('randomForest')
install.packages('dummies')
install.packages('Boruta')
install.packages('ROCR')
install.packages('glmnet')
install.packages('caret')
install.packages('e1071')
install.packages('corrplot')
install.packages('dplyr')
install.packages('xgboost')


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
library(xgboost)

path <- '~/Desktop/DePaul Compitit/NationalCompetite/data/features'
setwd(path)
###############################################################################################################
################################  MODELING  USING ALL GOOD BORUTA FEATURES#####################################
###############################################################################################################
################################                001150                    #####################################
###############################################################################################################

traindata <- read.csv('train_Boruta001150.csv',sep=',',header=TRUE,row.names = NULL)[,-1]
#View(traindata)
validata <- read.csv('validation_Boruta001150.csv',sep=',',header=TRUE,row.names = NULL)[,-1]
#View(validata)
traindatafull <- read.csv('train_Borutaall001150.csv',sep=',',header=TRUE)[,-1]
validatafull <- read.csv('validation_Borutaall001150.csv',sep=',',header=TRUE)[,-1]
#FIXED Y
traindata$Active_Customer <- as.factor(traindata$Active_Customer)
validata$Active_Customer <- as.factor(validata$Active_Customer)
traindatafull$Active_Customer <- as.factor(traindatafull$Active_Customer)
validatafull$Active_Customer <- as.factor(validatafull$Active_Customer)

#################################################################
################  BENCHEN MRK RANDOMFOREST  #####################
#################################################################

#RandomForest trees
# With ntree = 100 we have Accuracy of 0.6669 with 95%CI[0.6558,0.6779]
# Accuracy:0.6681  Sensitivity:0.6433  Specificity 0.6994
# With ntree = 200 we have Accuracy of 0.6703 with 95%CI[0.6592,0.6813]
# Accuracy:0.6703  Sensitivity:0.6438  Specificity 0.7047
# With ntree = 400 we have Accuracy of 0.6701 with 95%CI[0.659,0.681]
# Accuracy:0.6701  Sensitivity: 0.6438 Specificity:0.7038
# With ntree = 800 we have Accuracy of 0.6719 with 95%CI[0.6608,0.6829]
# Accuracy:0.6719  Sensitivity: 0.6457 Specificity:0.7056
RFfit <-randomForest(Active_Customer~.,data = traindata,ntree=800,importance=TRUE, proximity=TRUE)
RFpred <- predict(RFfit,validata)
resRF <- table(observed = validata$Active_Customer,predicted=RFpred)
confusionMatrix(resRF)

#RandomForest trees all
#With ntree = 400 we have Accuracy of 0.6686 with 95%CI[0.6575,0.6796]
#With ntree = 800 we have Accuracy of 0.6723 with 95%CI[0.6612,0.6796]
RFfit_all <-randomForest(Active_Customer~.,data = traindatafull,ntree=800,importance=TRUE, proximity=TRUE)
RFpred_all <- predict(RFfit_all,validatafull)
resRF_all <- table(observed = validatafull$Active_Customer,predicted=RFpred_all)
confusionMatrix(resRF_all)

#################Conclusion: Not using all features############################
###############################################################################################################
################################                001150                    #####################################
###############################################################################################################

###############################################################################################################
################################                005120                    #####################################
###############################################################################################################
traindata1<- read.csv('train_Boruta.csv',sep=',',header=TRUE,row.names = NULL)[,-1]
#View(traindata)
validata1 <- read.csv('validation_Boruta.csv',sep=',',header=TRUE,row.names = NULL)[,-1]
#View(validata)
traindatafull1 <- read.csv('train_Borutaall.csv',sep=',',header=TRUE)[,-1]
validatafull1 <- read.csv('validation_Borutaall.csv',sep=',',header=TRUE)[,-1]
#FIXED Y
traindata1$Active_Customer <- as.factor(traindata1$Active_Customer)
validata1$Active_Customer <- as.factor(validata1$Active_Customer)
traindatafull1$Active_Customer <- as.factor(traindatafull1$Active_Customer)
validatafull1$Active_Customer <- as.factor(validatafull1$Active_Customer)

#RF 005120
# With ntree = 400 we have Accuracy of 0.673 with 95%CI[0.662,0.684]
# Accuracy:0.673  Sensitivity: 0.6454 Specificity:0.7092
# With ntree = 600 we have Accuracy of 0.6706 with 95%CI[0.6595,0.6816]
# Accuracy:0.6706  Sensitivity: 0.6433 Specificity:0.7065
# With ntree = 500 we have Accuracy of 0.6711 with 95%CI[0.66,0.682]
# Accuracy:0.6711  Sensitivity: 0.6441 Specificity:0.7061
RFfit <-randomForest(Active_Customer~.,data = traindata1,ntree=500,importance=TRUE, proximity=TRUE)
RFpred <- predict(RFfit,validata1)
resRF <- table(observed = validata$Active_Customer,predicted=RFpred)
confusionMatrix(resRF)



#RandomForest trees all
#With ntree = 400 we have Accuracy of 0.6716 with 95%CI[0.6605,0.6826]
#With ntree = 600 we have Accuracy of 0.6725 with 95%CI[0.6614,0.6834]
RFfit_all <-randomForest(Active_Customer~.,data = traindatafull1,ntree=600,importance=TRUE, proximity=TRUE)
RFpred_all <- predict(RFfit_all,validatafull1)
resRF_all <- table(observed = validatafull$Active_Customer,predicted=RFpred_all)
confusionMatrix(resRF_all)
###############################################################################################################
################################                005120                    #####################################
###############################################################################################################



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
X_penlregre = as.matrix(traindata[,1:84])
Y_penlregre = traindata$Active_Customer
X_penlregre_validata = as.matrix(validata[,1:84])
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

#Using logit regression step back features to build RF
#Accuracy:0.673 95%CI[0.662,0.684]
RFfitbackward <-randomForest(formula(backwards),data = traindata,ntree=400,importance=TRUE, proximity=TRUE)
RFpredbackward <- predict(RFfitbackward,validata)
resRFbackward <- table(observed = validata$Active_Customer,predicted=RFpredbackward)
confusionMatrix(resRFbackward)


#################################################################
###################  XGBoosting Method ##########################
#################################################################
#####Original Features#######
X_xgb = as.matrix(traindata[,1:84])
Y_xgb = traindata$Active_Customer
X_xgb_validata = as.matrix(validata[,1:84])
Y_xgb_validata = validata$Active_Customer
xgbst_fit_originalfeature <- xgboost(data = X_xgb,label = Y_xgb,nrounds=30,max_depth=15,objective = 'binary:logistic')
Y_xgb_pred <- predict(xgbst_fit_originalfeature,X_xgb_validata)
Y_xgb_pred <- ifelse(Y_xgb_pred >0.5,1,0)
xgb_table <- table(observed =Y_xgb_validata,predicted=Y_xgb_pred)
confusionMatrix(xgb_table)

######Filtered Feature#####
train_backwardsdata <- as.matrix(traindata[,c(3,5,6,7,8,9,10,12,14,17,18,20,21,24,28,30,34,36,38,40,41,47,48,49,55,62,72,74,76,80,82)])
valid_backwardsdata <- as.matrix(validata[,c(3,5,6,7,8,9,10,12,14,17,18,20,21,24,28,30,34,36,38,40,41,47,48,49,55,62,72,74,76,80,82)])
xgb_backward <-xgboost(data = train_backwardsdata,label = Y_xgb,nrounds=30,max_depth=15,objective = 'binary:logistic')
Y_xgb_pred_backward <- predict(xgb_backward,valid_backwardsdata)
Y_xgb_pred_backward <- ifelse(Y_xgb_pred_backward >0.5,1,0)
xgb_backward_table <- table(observed = Y_xgb_validata,predicted=Y_xgb_pred_backward)
confusionMatrix(xgb_backward_table)

###############################################################################################################
# Random Forest performs better. Feature would be further filtered. #
###############################################################################################################