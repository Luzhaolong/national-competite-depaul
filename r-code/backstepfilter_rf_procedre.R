library(corrplot)
####### Readdata######
path <- '~/Desktop/DePaul Compitit/NationalCompetite/data/features'
setwd(path)
traindata <- read.csv('train_Boruta001150.csv',sep=',',header=TRUE,row.names = NULL)[,-1]
validata <- read.csv('validation_Boruta001150.csv',sep=',',header=TRUE,row.names = NULL)[,-1]
#FIXED Y
traindata$Active_Customer <- as.factor(traindata$Active_Customer)
validata$Active_Customer <- as.factor(validata$Active_Customer)
##Backstep feature selection
train_backwardsdata <- traindata[,c(3,5,6,7,8,9,10,12,14,17,18,20,21,24,28,30,34,36,38,40,41,47,48,49,55,62,72,74,76,80,82,85)]
valid_backwardsdata <- validata[,c(3,5,6,7,8,9,10,12,14,17,18,20,21,24,28,30,34,36,38,40,41,47,48,49,55,62,72,74,76,80,82,85)]


#Backstep_RF#
#ntree = 600 accuracy = 0.6712#
#ntree = 800 accuracy = 0.6733#
#ntree = 1000 accuracy = 0.6705#
RFfit <-randomForest(Active_Customer~.,data = train_backwardsdata,ntree=1000,importance=TRUE, proximity=TRUE)
RFpred <- predict(RFfit,valid_backwardsdata)
resRF <- table(observed = valid_backwardsdata$Active_Customer,predicted=RFpred)
confusionMatrix(resRF)

##############Further feature selection ###############
train_backwardsdata_cor <- train_backwardsdata[,-1]
cr <- cor(train_backwardsdata_cor[sapply(train_backwardsdata_cor, is.numeric)])
corrplot(cr,type = 'lower')
corrplot.mixed(cr)
