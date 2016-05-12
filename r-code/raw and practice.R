install.packages('mice')
library(VIM)
library(mice)
path <- '~/Desktop/DePaul Compitit'
setwd(path)
traindata <- read.csv('Train.csv',sep=',',header=TRUE)
aggr(traindata)



for (i in 0:ncol(traindata)){
    missing <- sum(is.na(traindata[,i]))
    if (missing >0.01*nrow(traindata)){
        print (c(i,missing))
    }
}

for (i in 0:nrow(traindata)){
    missing <- sum(is.na(traindata[i,]))
    if (missing >0.2*ncol(traindata)){
        print (c(i,missing))
    }
}
counter = 0
for (i in 0:nrow(traindata)){
    missing <- sum(is.na(traindata[i,]))
    if (missing >0.6*ncol(traindata)){
        counter = counter+ 1
    }
}
counter

names(traindata)[45:208]
traindatafood <- traindata[,45:208]
for (i in 0:nrow(traindatafood)){
    missing <- sum(is.na(traindatafood[i,]))
    if (missing >0.9*ncol(traindatafood)){
        print (c(i,missing))
    }
}

View(traindatafood)`

traindata[is.na(traindata)]=0
sum(is.na(traindata))
dim(traindata)


traindata1 <- traindata
train_nomis <- na.omit(traindata1)
aggr(train_nomis)
dim(train_nomis)
