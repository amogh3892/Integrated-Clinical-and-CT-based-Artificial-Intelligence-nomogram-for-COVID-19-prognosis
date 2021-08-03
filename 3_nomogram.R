rm(list=ls())

library(pROC)
library(rms)
library(data.table)
library(glmnet)
library(ggplot2)

nomotrain = read.csv(file = '<Input path to train file with all clinical variables annd deep learning predictions>')
nomotest = read.csv(file = '<Input path to test file with all clinical variables annd deep learning predictions>')

ddist <- datadist(nomotrain); options(datadist='ddist')

# change the variables based on the column names 
fit1 <- lrm(Labels ~ AIP+LDH+PT+ALB+AST+LYM., data=nomotrain)
nomopred1 = predict(fit1,nomotest,type="fitted.ind")

nomotrainpred = predict(fit1,nomotrain,type="fitted.ind")

roc1 <- roc(nomotest$Labels, nomopred1)
auc(roc1)

ci.auc(roc1)

# plot nomogram 
nom <- nomogram(fit1, fun=function(x)1/(1+exp(-x)),fun.at=c(.001,.01,.05,seq(.1,.9,by=.1),.95,.99,.999),funlabel="Predictived Value")
plot(nom, xfrac=.20)
