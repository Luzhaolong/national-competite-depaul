install.packages('randomForest')
install.packages('dummies')
install.packages('Boruta')
install.packages('ROCR')
install.packages('glmnet')
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

###########################################################################################
##########################  PREPROCESSING DATA  ###########################################
########################    1.INPUT DATA               ####################################        
########################    2.CONVERT NUMERIC          ####################################
########################    3.TRAIN-VALIDATON SPLITE   ####################################
########################    4.FEATURE ENGINEERING      ####################################
###########################################################################################
#This file is data directly delete the NA values
path <- '~/Desktop/DePaul Compitit'
setwd(path)
traindata <- na.omit(read.csv('Train.csv',sep=',',header=TRUE,row.names = NULL))[,-1]
#Convert factor into numeric
#str(traindata)
traindata$Cust_status<- as.numeric(traindata$Cust_status)
traindata$Trans24<- as.numeric(traindata$Trans24)
traindata$Trans25<- as.numeric(traindata$Trans25)
traindata$Trans26<- as.numeric(traindata$Trans26)
traindata$Trans27<- as.numeric(traindata$Trans27)

#Train-Test split
smp_size <- floor(0.7 *nrow(traindata))
set.seed(123)
train_idx <- sample(seq_len(nrow(traindata)),size = smp_size)
train <- traindata[train_idx,]
validation <- traindata[-train_idx,]


################################################
################  BORUTA  ######################
################################################
Bor.DePaulCom <- Boruta(Active_Customer~.,data = train,doTrace = 2,pValue=0.05,maxRuns=150)
Bor.DePaulCom_TenFixed <- TentativeRoughFix(Bor.DePaulCom)
attStats(Bor.DePaulCom)
plotZHistory(Bor.DePaulCom)
#Features
getSelectedAttributes(Bor.DePaulCom)
getSelectedAttributes(Bor.DePaulCom_TenFixed)

#Fix I/O data and export data for save the selected features#
#Train#
train_fselect <- train[,names(train)%in%getSelectedAttributes(Bor.DePaulCom)]
train_fselectall <- train[,names(train)%in%getSelectedAttributes(Bor.DePaulCom_TenFixed)]
train_fselect$Active_Customer <- train$Active_Customer
train_fselectall$Active_Customer <- train$Active_Customer
write.csv(train_fselect,file = 'train_Boruta.csv')
write.csv(train_fselectall,file = 'train_Borutaall.csv')

#Validation#
validation_fselect <- validation[,names(validation)%in%getSelectedAttributes(Bor.DePaulCom)]
validation_fselectall <- validation[,names(validation)%in%getSelectedAttributes(Bor.DePaulCom_TenFixed)]
validation_fselect$Active_Customer <- validation$Active_Customer
validation_fselectall$Active_Customer <- validation$Active_Customer
write.csv(validation_fselect,file = 'validation_Boruta.csv')
write.csv(validation_fselectall,file = 'validation_Borutaall.csv')
###############################################################################################################



###############################################################################################################
################################  MODELING  USING ALL GOOD BORUTA FEATURES#####################################
###############################################################################################################
traindata <- read.csv('train_Boruta.csv',sep=',',header=TRUE,row.names = NULL)[,-1]
View(traindata)
validata <- read.csv('validation_Boruta.csv',sep=',',header=TRUE,row.names = NULL)[,-1]
View(validata)
traindatafull <- read.csv('train_Borutaall.csv',sep=',',header=TRUE)[,-1]
validata <- read.csv('validation_Borutaall.csv',sep=',',header=TRUE)[,-1]

#FIXED Y
traindata$Active_Customer <- as.factor(traindata$Active_Customer)
validata$Active_Customer <- as.factor(validata$Active_Customer)

#################################################################
################  BENCHEN MRK RANDOMFOREST  #####################
#################################################################

#RandomForest trees
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

#Using Lasso to regularization the regression and finding the gradient descent parameter
X_penlregre = as.matrix(traindata[,1:83])
Y_penlregre = traindata$Active_Customer
X_penlregre_validata = as.matrix(validata[,1:83])
Y_penlregre_validata = validata$Active_Customer

lasso_logrgfit <- glmnet(X_penlregre,Y_penlregre,family='binomial',alpha=1)
summary(lasso_logrgfit)
lasso_log <- predict(lasso_logrgfit,X_penlregre_validata,type='response')
lasso_logpredreslt <- ifelse(lasso_log >0.5,1,0)
micClassificationerror_lasso <- mean(lasso_logpredreslt != validata$Active_Customer)
print(paste('Accuracy',1-micClassificationerror_lasso))
summary(lasso_logrgfit)
plot.glmnet(lasso_logrgfit)

ridge_logrgfit <- glmnet(X_penlregre,Y_penlregre,family='binomial',alpha=0)
ridge_log <- predict(ridge_logrgfit,X_penlregre_validata,type='response')
ridge_logpredreslt <- ifelse(ridge_log >0.5,1,0)
micClassificationerror_ridge <- mean(ridge_logpredreslt != validata$Active_Customer)
print(paste('Accuracy',1-micClassificationerror_ridge))
summary(lasso_logrgfit)
plot.glmnet(ridge_logrgfit)
#The following function may have teh problem of multicolinearity
#logitpred <- predict(logrgfit,validata)

backwards = step(logrgfit)
summary(backwards)

##########SVM########