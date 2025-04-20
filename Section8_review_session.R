# --------------------------------------------------------
# ST4061 / ST6041 / ST6042
# 2024-25
# Eric Wolsztynski
# Review session 1
# --------------------------------------------------------

rm(list=ls())

library(caret)
library(tree)
library(randomForest)

# -------------------------------------------------
# S2022 Q2

# Prepare the dataset:
dat.nona = na.omit(airquality)
dat = airquality
dat.nona$Month = as.factor(dat.nona$Month)
dat.nona$Day = as.factor(dat.nona$Day)
dat$Month = as.factor(dat$Month)
dat$Day = as.factor(dat$Day)

# (1) Identify two clear associations between the variables in the dataset. Use at least two relevant outputs as evidence of these associations.

round(cor(na.omit(dat)),3)
round(cor(na.omit(airquality)),3)
pairs(airquality)
boxplot(Wind~Month,data=airquality)
boxplot(Temp~Month,data=airquality)
boxplot(Ozone~Month,data=airquality)
cor(airquality$Ozone,airquality$Temp,"pairwise")
# chisq.test(airquality$Wind,airquality$Temp)
qqplot(scale(airquality$Wind),scale(airquality$Temp))
abline(a=0,b=1)
# Ozone ~ Temp (rho=0.699) 
# Temp ~ Month

# bonus (EDA)
library(DataExplorer)
# overall
introduce(dat)
plot_intro(dat)
plot_str(dat)
plot_missing(dat)
# distributions
plot_bar(dat)
plot_bar(dat,with="Ozone")
plot_boxplot(dat,by="Solar.R")
plot_histogram(dat)
plot_qq(dat)
plot_qq(dat,by="Ozone")
dat2 = dat
dat2$Solar.R = sqrt(dat2$Solar.R+1)
plot_qq(dat2)
# correlation analysis
plot_correlation(dat[,1:4])
cor(na.omit(dat[,1:4]))
plot_correlation(na.omit(dat[,1:4]))
plot_correlation(na.omit(dat))

# (2) Comment on your findings in (1) in terms of data imputation strategies for this dataset.
# - Ozone has many missing values, its association with 
# Temp (at least) should be taken into account for imputation.
# - Less important "patterns" for Solar.R, could use Temp 
# and/or Wind, or could use overall median Solar.R value

# (3) Impute Solar.R

is = which(is.na(dat$Solar.R))
dat$Solar.R[is] = median(dat$Solar.R[-is])
glm1.fit = glm(Wind~., data=dat)
sum((glm1.fit$residuals)^2)

# (4) Perform imputation of Ozone using a quadratic regression of Ozone onto Temp from na.omit(airquality).

is = which(is.na(dat$Ozone))
qmo = lm(Ozone~Temp+I(Temp^2), data=na.omit(airquality))
Ozq = predict(qmo, newdata=airquality[is,])
dat$Ozone[is] = Ozq
glm3.fit = glm(Wind~., data=dat)
sum((glm3.fit$residuals)^2)
plot(dat$Ozone,dat$Temp)
points(airquality$Ozone[-is],airquality$Temp[-is],pch=20)

# (5) Fit an unpruned tree 

tree.fit = tree(Wind~., data=dat)
summary(tree.fit)
summary(tree.fit)$used
plot(tree.fit)
text(tree.fit)
sum(summary(tree.fit)$residuals^2)

# (6) Prune the tree

pruned.fit = prune.tree(tree.fit, best=10)
summary(pruned.fit)
summary(pruned.fit)$used
sum(summary(pruned.fit)$residuals^2)

# -------------------------------------------------
# S2022 Q3

rm(list=ls())

library(ISLR) # for the data
library(randomForest) 
library(gbm)

xtrain = Khan$xtrain
xtest = Khan$xtest
ytrain = as.factor(Khan$ytrain)
ytest = as.factor(Khan$ytest)

round(prop.table(table(ytrain)),3)
round(prop.table(table(ytest)),3)
round(table(ytrain)/length(ytrain),3)*100
round(table(ytest)/length(ytest),3)*100

set.seed(4061)
rfo = randomForest(xtrain,ytrain)
rfo

rfpr = predict(rfo, xtest, type="prob")
rfp = predict(rfo, xtest)
tb = table(rfp,ytest)
sum(diag(tb))/sum(tb)

is = which(rfo$importance>0.4)
rownames(rfo$importance)[is]
# hist(rfo$importance[is])
# barplot((rfo$importance[is,]), las=2)

#USe multinomial because we have a multinomial problem. We use bernoulli for binary and gaussian for regression, or poisson, laplace, quantile etc
gb.out = gbm(ytrain~., data=as.data.frame(xtrain), distribution='multinomial') 
gb.fitted = predict(gb.out, n.trees=gb.out$n.trees) 
gb.pred = predict(gb.out, as.data.frame(xtest), n.trees=gb.out$n.trees)
gbp = apply(gb.pred,1,which.max)
tb = table(ytest, gbp)
sum(diag(tb))/sum(tb)
