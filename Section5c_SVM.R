# --------------------------------------------------------
# ST4061 / ST6041 / ST6042
# 2024-25
# Eric Wolsztynski
# ...
# Exercises Section 5: SVM
# --------------------------------------------------------

rm(list=ls())

library(randomForest)
library(class)
library(pROC)
library(e1071)
library(caret)
library(ISLR)

###############################################################
### Exercise 1: using SVM... (High Carseat sales dataset)
###############################################################

require(ISLR) # contains the dataset
# Recode response variable so as to make it a classification problem
High = ifelse(Carseats$Sales<=8, "No", "Yes")
CS = data.frame(Carseats, High)
CS$Sales = NULL
x = CS
x$High = NULL
y = CS$High

# split the data into train+test:
n = nrow(CS)
set.seed(4061)
i.train = sample(1:n, 350) 
x.train = x[i.train,]
x.test = x[-i.train,]
y.train = y[i.train]
y.test = y[-i.train]

class(x.train)
class(y.train)
# ?svm # in e1071

# (3) Explain why this does not work:
svmo = svm(x.train, y.train, kernel='polynomial') 

# >> The problem is the presence of categorical variables in 
# the dataset. They must be "recoded" into numerical variables 
# for svm() to analyse their spatial contribution.

# (4) Carry out the appropriate fix from your conclusion from (a). 
# Then, fit two SVM models, one using a linear kernel, the other 
# a polynomial kernel. Compare the two appropriately.

NC = ncol(x)
# x = x[,-c(NC-1,NC)] # take out the last two columns (predictors)
xm = model.matrix(y~.+0, data=x)
xm.train = xm[i.train,]
xm.test = xm[-i.train,]
y.train = as.factor(y.train) # so that svm knows it's classification
svmo.lin = svm(xm.train, y.train, kernel='linear') 
svmo.pol = svm(xm.train, y.train, kernel='polynomial') 

svmy.lin = fitted(svmo.lin)
svmy.pol = fitted(svmo.pol)
table(y.train, svmy.lin)
table(y.train, svmy.pol)
pairs(xm.train[,c(1:6)], pch=20, col=c(1,4)[y.train], cex=2)
# (5) Comparison...
# * visual (there are better ways of visualising!):
par(mfrow=c(1,3))
yl = as.integer(y=="Yes")+1
plot(apply(xm,2,scale), pch=c(15,20)[yl], col=c(1,4)[yl], 
	cex=c(1.2,2)[yl], main="The data")
#
plot(apply(xm.train,2,scale), pch=c(15,20)[y.train], col=c(1,4)[y.train], 
	cex=1, main="linear")
points(svmo.lin$SV, pch=5, col=2, cex=1.2)
#
plot(apply(xm.train,2,scale), pch=c(15,20)[y.train], col=c(1,4)[y.train], 
	cex=1, main="polynomial")
points(svmo.pol$SV, pch=5, col=2, cex=1.2)
svmo.lin
svmo.pol

# * in terms of training fit:
svmy.lin = fitted(svmo.lin)
svmy.pol = fitted(svmo.pol)
table(y.train, svmy.lin)
table(y.train, svmy.pol)

# * test error:
pred.lin = predict(svmo.lin, newdata=xm.test, probability=TRUE)
pred.pol = predict(svmo.pol, newdata=xm.test)
# ... the above does not work well:
summary(pred.lin)
# --> these are not probabilities! That's because we need to specify
# ", probability=TRUE" also when fitting the SVM, in order to enable 
# probabilities to be computed and returned...
# SO IF WE WANT TO GENERATE TEST-SET PREDICTIONS, THIS IS THE WAY:
svmo.lin = svm(xm.train, y.train, kernel='linear', probability=TRUE) 
svmo.pol = svm(xm.train, y.train, kernel='polynomial', probability=TRUE) 
pred.lin = predict(svmo.lin, newdata=xm.test, probability=TRUE)
pred.pol = predict(svmo.pol, newdata=xm.test, probability=TRUE)

y.test = as.factor(y.test)
confusionMatrix(data=pred.lin, reference=y.test, positive='Yes')
confusionMatrix(data=pred.pol, reference=y.test, positive='Yes')
# * AUC (we need to extract P(Y=1|X))
p.lin = attributes(pred.lin)$probabilities[,2]
p.pol = attributes(pred.pol)$probabilities[,2]
y.test = as.factor(y.test)
roc(response=y.test, predictor=p.lin)$auc
roc(response=y.test, predictor=p.pol)$auc

###############################################################
### Exercise 2: 3-class problem (iris dataset)
###############################################################

x = iris
x$Species = NULL
y = iris$Species

set.seed(4061)
n = nrow(x)
i.train = sample(1:n, 100) 
x.train = x[i.train,]
x.test = x[-i.train,]
y.train = y[i.train]
y.test = y[-i.train]

# (a) 
plot(x.train[,1:2], pch=20, col=c(1,2,4)[as.numeric(y.train)], cex=2)

# (b)
dat = data.frame(x.train, y=as.factor(y.train))
svmo.lin = svm(y~., data=dat, kernel='linear') 
svmo.pol = svm(y~., data=dat, kernel='polynomial') 
svmo.rad = svm(y~., data=dat, kernel='radial') 
#
# number of support vectors:
summary(svmo.lin)
summary(svmo.pol)
summary(svmo.rad)
#
# test error:
pred.lin = predict(svmo.lin, newdata=x.test)
pred.pol = predict(svmo.pol, newdata=x.test)
pred.rad = predict(svmo.rad, newdata=x.test)
cm.lin = confusionMatrix(y.test, pred.lin)
cm.pol = confusionMatrix(y.test, pred.pol)
cm.rad = confusionMatrix(y.test, pred.rad)
c(cm.lin$overall[1], cm.pol$overall[1], cm.rad$overall[1])

# (c) tune (via cross-validation)...
set.seed(4061)
svm.tune = e1071::tune(svm, train.x=x.train, train.y=y.train,
 	kernel='radial',
	ranges=list(cost=10^(-2:2), gamma=c(0.5,1,1.5,2)))
print(svm.tune)
names(svm.tune)

# retrieve optimal hyper-parameters
bp = svm.tune$best.parameters

# use these to obtain final SVM fit:
svmo.rad.tuned = svm(y~., data=dat, kernel='radial', 
	cost=bp$cost, gamma=bp$gamma) 
summary(svmo.rad)
summary(svmo.rad.tuned)

# test set predictions from tuned SVM model:
pred.rad.tuned = predict(svmo.rad.tuned, newdata=x.test)
cm.rad.tuned = confusionMatrix(y.test, pred.rad.tuned)
c(cm.rad$overall[1], cm.rad.tuned$overall[1])
# so maybe not an exact science!?
# ... in fact these performances are comparable, bear in mind CV assessment is itself subject to variability...


###############################################################
### Exercise 3: SVM using caret
###############################################################

# Set up the data (take a subset of the Hitters dataset)
data(Hitters)
Hitters = na.omit(Hitters)
dat = Hitters
n = nrow(dat)
NC = ncol(dat)
# Change into a classification problem:
dat$Salary = as.factor(ifelse(Hitters$Salary>median(Hitters$Salary),
						"High","Low"))

# Data partition
set.seed(4061)
itrain = sample(1:n, size=round(.7*n))
dat.train = dat[itrain,]
dat.validation = dat[-itrain,] # independent validation set
x = dat.train # training set
x$Salary = NULL
y = as.factor(dat.train$Salary)

### Random forest
rf.out = caret::train(Salary~., data=dat.train, method='rf')
rf.pred = predict(rf.out, dat.validation)
rf.cm = confusionMatrix(reference=dat.validation$Salary, data=rf.pred, mode="everything")

### SVM (linear)
svm.out = caret::train(Salary~., data=dat.train, 
	method="svmLinear")
svm.pred = predict(svm.out, dat.validation)
svm.cm = confusionMatrix(reference=dat.validation$Salary, data=svm.pred, mode="everything")

### SVM (radial)
modelLookup('svmRadial')
svmR.out = caret::train(Salary~., data=dat.train, method="svmRadial")
svmR.pred = predict(svmR.out, dat.validation)
svmR.cm = confusionMatrix(reference=dat.validation$Salary, data=svmR.pred, mode="everything")

perf = rbind(rf.cm$overall, svm.cm$overall, svmR.cm$overall)
row.names(perf) = c("RF","SVM.linear","SVM.radial")
round(perf, 4)

perf = cbind(rf.cm$overall, svm.cm$overall, svmR.cm$overall)
colnames(perf) = c("RF","SVM.linear","SVM.radial")
round(perf, 4)

###############################################################
### Exercise 4: SVM-based regression
###############################################################

x = iris
x$Sepal.Length = NULL
y = iris$Sepal.Length
pairs(iris[,1:4])

set.seed(4061)
n = nrow(x)
i.train = sample(1:n, 100) 
x.train = x[i.train,]
x.test = x[-i.train,]
y.train = y[i.train]
y.test = y[-i.train]
dat.train = cbind(x.train,y=y.train)

# specify statistical training settings:
ctrl = caret::trainControl(method='cv')
# perform statistical training:
svm.o = caret::train(y~., data=dat.train, method="svmLinear", 
			trControl=ctrl)
# compute test set predictions:
svm.p = predict(svm.o, newdata=x.test)
# and corresponding MSE:
mean( (y.test-svm.p)^2 )

par(pty='s') # makes or a square plot box
rr = range(c(y.test, svm.p))
plot(y.test, svm.p, pch=20, 
	xlab="true values", ylab="predicted values",
	xlim=rr,ylim=rr)
abline(a=0,b=1)

