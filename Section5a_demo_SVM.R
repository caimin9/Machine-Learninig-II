# --------------------------------------------------------
# ST4061 / ST6041 / ST6042
# 2024-25
# Eric Wolsztynski
# ...
# Section 5: demo code for SVM
# --------------------------------------------------------

rm(list=ls())  # clear out running environment

library(randomForest)
library(class)
library(pROC)
library(e1071)

### Example using SVM on the "best-student" data

# simulate data:
set.seed(1)
n = 100
mark = rnorm(n, m=50, sd=10)
choc = rnorm(n, m=60, sd=5)
summary(mark)
summary(choc)
int = 10
a = 2
b = 4
# rating of students on basis of their marks and level 
# of appreciation of chocolate:
mod = int + a*mark + b*choc   # values for true model
z = rnorm(n,s=8)  # additive noise
obs = mod + z
y = as.factor(ifelse(obs>350,1,0))  # classification data
plot(mod, obs, xlab="model", ylab="observations", pch=20, cex=2)
par(font=2,font.axis=2,font.lab=2)
plot(mark, choc, xlab="x1 (mark)", ylab="x2 (choc)", 
	pch=20, col=c(2,4)[as.numeric(y)], cex=2)
legend("topright", box.col=8, bty='n', 
		legend=c("good student","better student"),
		pch=15, col=c(2,4))
table(y)

set.seed(1)
# split the data into train+test (50%-50%):
x = data.frame(mark,choc)
i.train = sample(1:n, 50) 
x.train = x[i.train,]
x.test = x[-i.train,]
y.train = y[i.train]
y.test = y[-i.train]

class(x.train)
class(y.train)
xm = as.matrix(x.train)
# fit an SVM as follows:
# ?svm # in e1071
set.seed(1)
svmo = svm(xm, y.train, kernel='polynomial') 
svmo
names(svmo)

# cf. ?svm:
# "Parameters of SVM-models usually must be tuned to yield sensible results!"
# one can tune the model as follows:
set.seed(1)
svm.tune = e1071::tune(svm, train.x=x.train, train.y=y.train,
	kernel='polynomial',
	ranges=list(cost=10^(-2:4), gamma=c(0.25,0.5,1,1.5,2)))
svm.tune
svm.tune$best.parameters

# then fit final SVM for optimal parameters:
svmo.final = svm(xm, y.train, kernel='polynomial',
	gamma=svm.tune$best.parameters$gamma,
	cost=svm.tune$best.parameters$cost) 

# corresponding confusion matrices:
table(svmo$fitted,y.train)
table(svmo.final$fitted,y.train)

# we can also use caret for easy comparison:
library(caret)
caret::confusionMatrix(svmo$fitted,y.train)$overall[1]
caret::confusionMatrix(svmo.final$fitted,y.train)$overall[1]

# assessing model fit to training data
identical(fitted(svmo), svmo$fitted)

# to identify support vectors: 
# either svmo$index (indices), or svmo$SV (coordinates)
length(svmo$index)
length(svmo.final$index)
# visualize:
plot(x.train, pch=20, col=c(1,2)[y.train], cex=2)
points(svmo$SV, pch=14, col=4, cex=2) # explain why this does not work!?

# apply scaling to dataset to see SV's:
plot(apply(x.train,2,scale), pch=20, col=c(1,2)[y.train], cex=2)
points(svmo$SV, pch=14, col=4, cex=2)
points(svmo.final$SV, pch=5, col=4, cex=2)

# If you want to use predict(), use a formula-type
# expression when calling svm(). Because of this,
# we re-shape our dataset:
dat.train = data.frame(x=x.train, y=y.train)
dat.test = data.frame(x=x.test)

# decision boundary visualization:
svmo = svm(y~., data=dat.train)
plot(svmo, dat.train, 
	svSymbol = 15, dataSymbol = 'o',
	col=c('cyan','pink')) # this is plot.svm()
svmo.final = svm(y~., data=dat.train, kernel='polynomial',
	gamma=svm.tune$best.parameters$gamma,
	cost=svm.tune$best.parameters$cost) 
plot(svmo.final, dat.train, 
	svSymbol = 15, dataSymbol = 'o',
	col=c('cyan','pink')) # this is plot.svm()

# How to generate predictions from SVM fit: 
svmo = svm(y~., data=dat.train,
	kernel='polynomial',
	gamma=svm.tune$best.parameters$gamma,
	cost=svm.tune$best.parameters$cost)
# Note that if we need probabilities P(Y=1|X)
# (for ex to calculate ROC+AUC),
# we need to set 'probability=TRUE' also in 
# fitting the SVM model:
svmo = svm(y~., data=dat.train, probability=TRUE,
	kernel='polynomial',
	gamma=svm.tune$best.parameters$gamma,
	cost=svm.tune$best.parameters$cost) 
svmp = predict(svmo, newdata=dat.test, probability=TRUE)
roc.svm = roc(y.test, attributes(svmp)$probabilities[,2])
roc.svm$auc
plot(roc.svm)

# compare with RF:
rf = randomForest(y~., data=dat.train)
rfp = predict(rf, dat.test, type='prob')
roc.rf = roc(y.test, rfp[,2])
roc.rf$auc
plot(roc.svm)
par(new=TRUE)
plot(roc.rf, col='yellow')
legend("bottomright", bty='n',
		legend=c("RF","SVM"),
		lty=1, lwd=3, col=c('yellow',1))
		
