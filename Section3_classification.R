# --------------------------------------------------------
# ST4061 / ST6041 / ST6042
# 2024-25
# Eric Wolsztynski
# ...
# Exercises Section 3: Classification Exercises
# --------------------------------------------------------

library(class) # contains knn()
library(MASS)  # to have lda()
library(car)
library(ISLR) # contains the datasets
library(pROC) 
library(caret)

###############################################################
### Exercise 1: kNN on iris data
###############################################################

set.seed(1)
# shuffle dataset first:
z = iris[sample(1:nrow(iris)),] 

# Here we focus on sepal information only
plot(z[,1:2], col=c(1,2,4)[z[,5]], 
	pch=20, cex=2)
x = z[,1:2] # sepal variables only
y = z$Species

# Here's how to use the knn() classifier:
K = 5
n = nrow(x)
# split the data into train+test:
i.train = sample(1:n, 100)
x.train = x[i.train,]
x.test = x[-i.train,]
y.train = y[i.train]
y.test = y[-i.train]
ko = knn(x.train, x.test, y.train, K)
tb = table(ko, y.test)
tb
1 - sum(diag(tb)) / sum(tb) 
# overall misclassification error rate

confusionMatrix(data=ko, reference=y.test)

# Build a loop around that to find best k:
# (NB: assess effect of various k-values 
# on the same data-split)
Kmax = 30
acc = numeric(Kmax)
for(k in 1:Kmax){
	ko = knn(x.train, x.test, y.train, k)
	tb = table(ko, y.test)
	acc[k] = sum(diag(tb)) / sum(tb)	
}
plot(1-acc, pch=20, t='b', xlab='k')

# !!! ROC analysis of knn() output !!!
# WARNING: 
# ?knn:
# "the proportion of the votes for the winning class 
# are returned as attribute prob"
# in other words:
# attributes(knn(x.train, x.test, y.train, k, prob=TRUE))$prob
# does not return Pr(Y=1 | X), but Pr(Y = Y_hat | X)...

# Need to "hack" if we want Pr(Y=1 | X)...
ko = knn(x.train, x.test, y.train, k)
ko = knn(x.train, x.test, y.train, k, prob=TRUE)
kp = attributes(ko)$prob
# (not so easy for a 3-class problem)

############################################################### 
### Exercise 2: GLM on 2-class iris data
############################################################### 

n = nrow(iris)
is = sample(1:n, size=n, replace=FALSE)
dat = iris[is,-c(3,4)] # shuffled version of the original set
# recode into 2-class problem:
dat$is.virginica = as.numeric(dat$Species=="virginica") 
dat$Species = NULL # "remove" this component
names(dat)

is = 1:100 # training set
fit = glm(is.virginica~., data=dat, subset=is,
				family=binomial(logit))
pred = predict(fit, newdata=dat[-is,], type="response")
y.test = dat$is.virginica[-is] # true test set classes
boxplot(pred~y.test, names=c("other","virginica"))
# boxplot(pred~y.test, names=c("cancer absent","cancer present"))
abline(h=0.5, col=3)
abline(h=0.1, col=4)

preds = as.factor(as.numeric(pred>.5))
confusionMatrix(data=preds, reference=as.factor(y.test))
# watch out for "positive class"!
o = confusionMatrix(data=preds, reference=as.factor(y.test),
	positive="1")
o
# alternative option:
# preds= relevel(preds,"1")

names(o)
o$byClass[1]

# for varying cut-off (ie threshold) values, compute corresponding 
# predicted labels, and corresponding confusion matrix:
err = NULL
for(cut.off in seq(.1, .9, by=.1)){
	pred.y = as.numeric(pred>cut.off)
	# either overall accuracy:
	# tb = table(pred.y, y.test)
	# err = c(err, (1-sum(diag(tb))/sum(tb)))
	#
	o = confusionMatrix(data=as.factor(pred.y), 
					reference=as.factor(y.test),
	positive="1")
	err = c(err,o$byClass[1])
}
plot(seq(.1, .9, by=.1), err, t='b')

###############################################################
### Exercise 3: LDA assumptions 
############################################################### 

## (1) 2-class classification problem

dat = iris
dat$Species = as.factor(ifelse(iris$Species=="virginica",1,0))
# to recode cleanly, you could use for instance:
dat$Species = car::recode(dat$Species, "0='other'; 1='virginica'")
# or:
# levels(dat$Species) = c("other","virginica")
#
par(mfrow=c(1,2))
plot(iris[,1:2], pch=20, col=c(1,2,4)[iris$Species], cex=2)
legend("topright",col=c(1,2,4),
	legend=levels(iris$Species),
	pch=20, bty='n')
plot(dat[,1:2], pch=20, col=c(1,4)[dat$Species], cex=2)
legend("topright",col=c(1,4),
	legend=levels(dat$Species),
	pch=20, bty='n')

## Explore distribution of predictors:
# boxplots seem relatively symmetric:
par(mfrow=c(2,2))
for(j in 1:4){ 
	boxplot(dat[,j]~dat$Species, 
	      ylab='predictor',
				col=c('cyan','pink'), 
				main=names(dat)[j])
}
# but we'd rather check for Normality more specifically:
par(mfrow=c(2,4), font.lab=2, cex=1.2)
for(j in 1:4){ 
	hist(dat[which(dat$Species=='other'),j], col='cyan', 
	      xlab='predictor for class other',
				main=names(dat)[j])
	hist(dat[which(dat$Species!='other'),j], col='pink', 
	     xlab='predictor for class virginica',
	     main=names(dat)[j])
}

# note: 
library(DataExplorer)
# pdf("my_file.pdf")
plot_histogram(dat[,1:4])
# dev.off()
plot_histogram(dat[which(dat$Species=='other'),1:4])
plot_histogram(dat[which(dat$Species=='virginica'),1:4])
plot_boxplot(dat,by='Species')

# could also use QQ-plots:
par(mfrow=c(2,4), cex=1.2)
for(j in 1:4){ 
	x.other = dat[which(dat$Species=='other'),j]
	qqnorm(x.other, pch=20, col='cyan', 
				main=names(dat)[j])
	abline(a=mean(x.other), b=sd(x.other))
	x.virginica = dat[which(dat$Species!='other'),j]
	qqnorm(x.virginica, pch=20, col='pink', 
				main=names(dat)[j])
	abline(a=mean(x.virginica), b=sd(x.virginica))
}
# So what do you think?

## Check initial assumption of equality of variances:
# Bartlett's test with H0: all variances are equal
for(j in 1:4){
	print( bartlett.test(dat[,j]~dat$Species)$p.value )
}

# Shapiro's test with H0: the distribution is Normal
for(j in 1:4){
  print( shapiro.test(dat[which(dat$Species=='virginica'),j])$p.value ) 
}
for(j in 1:4){
  print( shapiro.test(dat[which(dat$Species=='other'),j])$p.value ) 
}

## Fit LDA model to this dataset and check accuracy:
lda.o = lda(Species~., data=dat)
(lda.o)
# can we track some of the values in that summary?
table(dat$Species)/nrow(dat)
rbind(
    apply(dat[which(dat$Species=='other'),1:4], 2, mean),
    apply(dat[which(dat$Species=='virginica'),1:4], 2, mean)
)
# what about the coefficients of linear discriminants?
x = as.matrix(dat[,1:4])
proj = x %*% lda.o$scaling
plot(proj, pch=20, col=dat$Species, cex=2)
# little hack to recover the fitted values quickly
predo = predict(lda.o, newdata=dat)
y = predo$x
plot(proj, y)
plot(y, predo$posterior[,2])
boxplot(y ~ (predo$posterior[,2]>.5))
boxplot(proj ~ (predo$posterior[,2]>.5))
# NB: The way these coefs is calculated follows the MANOVA approach

# popular hack to recover the fitted values:
fitted.values = predict(lda.o, newdata=dat)$class  
boxplot(y~dat$Species)
boxplot(proj~dat$Species)
(tb.2 = table(fitted.values, dat$Species))
sum(diag(tb.2)) / sum(tb.2)

## (2) 3-class classification problem

dat = iris
	
## Explore distribution of predictors:
# boxplots seem relatively symmetric:
par(mfrow=c(2,2))
# here's just a loop to save having to write 4 boxplot
# instructions with names by hand (being lazy often 
# makes for nicer code):
for(j in 1:4){ 
	boxplot(dat[,j]~dat$Species,
				xlab = 'Species',
				ylab = 'predictor',
				col=c('cyan','pink'), 
				main=names(dat)[j])
}
# but we'd rather check for Normality more specifically:
Ls = levels(dat$Species)
par(mfcol=c(3,4))
for(j in 1:4){ 
	hist(dat[which(dat$Species==Ls[1]),j], col='cyan', 
				main=names(dat)[j])
	hist(dat[which(dat$Species==Ls[2]),j], col='pink', 
				main=names(dat)[j])
	hist(dat[which(dat$Species==Ls[3]),j], col='green', 
				main=names(dat)[j])
}
# could also use QQ-plots:
par(mfcol=c(3,4))
for(j in 1:4){ 
	x1 = dat[which(dat$Species==Ls[1]),j]
	qqnorm(x1, pch=20, col='cyan', main=names(dat)[j])
	abline(a=mean(x1), b=sd(x1))
	x2 = dat[which(dat$Species==Ls[2]),j]
	qqnorm(x2, pch=20, col='pink', main=names(dat)[j])
	abline(a=mean(x2), b=sd(x2))
	x3 = dat[which(dat$Species==Ls[3]),j]
	qqnorm(x3, pch=20, col='green', main=names(dat)[j])
	abline(a=mean(x3), b=sd(x3))
}
# So what do you think now?

## Check initial assumption of equality of variances:
# Bartlett's test with H0: all variances are equal
print( bartlett.test(dat[,1]~dat$Species)$p.value )
print( bartlett.test(dat[,2]~dat$Species)$p.value )
print( bartlett.test(dat[,3]~dat$Species)$p.value )
print( bartlett.test(dat[,4]~dat$Species)$p.value )
## or in lazy mode:
for(j in 1:4){
	print( bartlett.test(dat[,j]~dat$Species)$p.value )
}

## Fit LDA model to this dataset and check accuracy:
lda.o = lda(Species~., data=dat)
(lda.o)
ftted.values = predict(lda.o, newdata=dat)$class
(tb.3 = table(ftted.values, dat$Species))
sum(diag(tb.3)) / sum(tb.3)

############################################################### 
### Exercise 4: LDA 
###############################################################

## (1) 2-class classification problem

dat = iris
dat$Species = as.factor(ifelse(iris$Species=="virginica",1,0))
levels(dat$Species) = c("other","virginica")

n = nrow(dat)
set.seed(4061)
dat = dat[sample(1:n),] # shuffle dataset

i.train = 1:100
dat.train = dat[i.train,]
dat.test = dat[-i.train,]
lda.o = lda(Species~., data=dat.train)
lda.p = predict(lda.o, newdata=dat.test)
names(lda.p)
(tb = table(lda.p$class, dat.test$Species))
sum(diag(tb))/sum(tb)

# QDA:
qda.o = qda(Species~., data=dat.train)
qda.p = predict(qda.o, newdata=dat.test)
(tb = table(qda.p$class, dat.test$Species))
sum(diag(tb))/sum(tb)

## (2) 3-class classification problem

dat = iris
n = nrow(dat)
set.seed(4061)
dat = dat[sample(1:n),]

i.train = 1:100
dat.train = dat[i.train,]
dat.test = dat[-i.train,]

# LDA:
lda.o = lda(Species~., data=dat.train)
lda.p = predict(lda.o, newdata=dat.test)
names(lda.p)
(tb = table(lda.p$class, dat.test$Species))
sum(diag(tb))/sum(tb)

# QDA:
qda.o = qda(Species~., data=dat.train)
qda.p = predict(qda.o, newdata=dat.test)
(tb = table(qda.p$class, dat.test$Species))
sum(diag(tb))/sum(tb)


############################################################### 
### Exercise 5: benchmarking
############################################################### 

## (1) benchmarking on unscaled data

set.seed(4061)
n = nrow(Default)
dat = Default[sample(1:n, n, replace=FALSE), ]

# get a random training sample containing 70% of original sample:
i.cv = sample(1:n, round(.7*n), replace=FALSE)
dat.cv = dat[i.cv,] # use this for CV (train+test)
dat.valid = dat[-i.cv,] # save this for later (after CV) (HOLD-OUT)

# tuning of the classifiers:
K.knn = 3 

# perform K-fold CV:
K = 10 
N = length(i.cv)
folds = cut(1:N, K, labels=FALSE)
acc.knn = acc.glm = acc.lda = acc.qda = numeric(K)
auc.knn = auc.glm = auc.lda = auc.qda = numeric(K)
#
for(k in 1:K){ # 10-fold CV loop
	# split into train and test samples:
	i.train	= which(folds!=k)
	dat.train = dat.cv[i.train, ]
	dat.test = dat.cv[-i.train, ]
	# adapt these sets for kNN:
	x.train = dat.train[,-1]
	y.train = dat.train[,1]
	x.test = dat.test[,-1]
	y.test = dat.test[,1]
	x.train[,1] = as.numeric(x.train[,1])
	x.test[,1] = as.numeric(x.test[,1])
	# train classifiers:
	knn.o = knn(x.train, x.test, y.train, K.knn)
	glm.o = glm(default~., data=dat.train, family=binomial(logit))
	lda.o = lda(default~., data=dat.train)
	qda.o = qda(default~., data=dat.train)
	# test classifiers:
	knn.p = knn.o
	glm.p = ( predict(glm.o, newdata=dat.test, type="response") > 0.5 )
	lda.p = predict(lda.o, newdata=dat.test)$class
	qda.p = predict(qda.o, newdata=dat.test)$class	
	tb.knn = table(knn.p, y.test)
	tb.glm = table(glm.p, y.test)
	tb.lda = table(lda.p, y.test)
	tb.qda = table(qda.p, y.test)
	# store prediction accuracies:
	acc.knn[k] = sum(diag(tb.knn)) / sum(tb.knn)
	acc.glm[k] = sum(diag(tb.glm)) / sum(tb.glm)
	acc.lda[k] = sum(diag(tb.lda)) / sum(tb.lda)
	acc.qda[k] = sum(diag(tb.qda)) / sum(tb.qda)
	#
	# ROC/AUC analysis:
	# WARNING: THIS IS NOT Pr(Y=1 | X), BUT Pr(Y = Y_hat | X):
	# knn.p = attributes(knn(x.train, x.test, y.train, K.knn, prob=TRUE))$prob
	glm.p = predict(glm.o, newdata=dat.test, type="response")
	lda.p = predict(lda.o, newdata=dat.test)$posterior[,2]
	qda.p = predict(qda.o, newdata=dat.test)$posterior[,2]
	# auc.knn[k] = roc(y.test, knn.p)$auc
	auc.glm[k] = roc(y.test, glm.p)$auc
	auc.lda[k] = roc(y.test, lda.p)$auc
	auc.qda[k] = roc(y.test, qda.p)$auc
}
boxplot(acc.knn, acc.glm, acc.lda, acc.qda,
	main="Overall CV prediction accuracy",
	names=c("kNN","GLM","LDA","QDA"))
boxplot(auc.glm, auc.lda, auc.qda,
	main="Overall CV AUC",
	names=c("GLM","LDA","QDA"))
	
##### Taking a closer look at performance
roc(y.test, glm.p)$auc
library(caret)
(tb = table(y.test, glm.p>.5))
pred = as.factor(glm.p>.5)
pred = car::recode(pred, "FALSE='No'; TRUE='Yes'")
caret::confusionMatrix(y.test, pred)
sum(diag(tb))/sum(tb)

##### Further exercises for you to do:
## adapt code to evaluate sensitivity and specificity
## add validation analysis... 
## repeat on scaled data... 


############################################################### 
### Exercise 6: benchmarking, again
############################################################### 

## (1) benchmarking on unscaled data

set.seed(4061)
n = nrow(Default)
dat = Default[sample(1:n, n, replace=FALSE), ]

# get a random training sample containing 70% of original sample:
i.cv = sample(1:n, round(.7*n), replace=FALSE)
x = dat.cv = dat[i.cv,] # use this for CV (train+test)
dat.valid = dat[-i.cv,] # save this for later (after CV)

# Recover ROC curve manually from whole set:

n = nrow(x)
acc = numeric(length(thrs))
sens = spec = numeric(length(thrs))
thrs = seq(.05,.95,by=.05)
for(ithr in 1:length(thrs)){
	thr = thrs[ithr]
	glmo = glm(default~., data=x, 
	          family=binomial)
	  tb = table(glmo$fitted.values>thr, x$default)
	  acc[ithr] = sum(diag(tb))/sum(tb)
	  #
	  # calculate sensitivity for a given threshold
	  sens[ithr] = ...
	  # calculate specificity for a given threshold
	  spec[ithr] = ...
	  # prediction
}	
plot(acc)
plot(spec, sens)

# Evaluate a cross-validated ROC curve manually:

n = nrow(x)
K = 10
train.acc = test.acc = matrix(NA, nrow=K, ncol=length(thrs))
folds = cut(1:n, K, labels=FALSE)
k = 1
thrs = seq(.05,.95,by=.05)
for(ithr in 1:length(thrs)){
	thr = thrs[ithr]
	for(k in 1:K){
	  itrain = which(folds!=k)
	  glmo = glm(default~., data=x, 
	          family=binomial,
	          subset=itrain)
	  tb = table(glmo$fitted.values>thr, x$default[itrain])
	  train.acc[k, ithr] = sum(diag(tb))/sum(tb)
	  #
	  # calculate sensitivity for a given threshold
	  # ...
	  # calculate specificity for a given threshold
	  # ...
	  # prediction
	  p.test = predict(glmo, x[-itrain,], type='response')
	  tb = table(p.test>thr, x$default[-itrain])
	  test.acc[k,ithr] = sum(diag(tb))/sum(tb)
	}
}	
boxplot(test.acc)
# warnings()
mean(train.acc)  
mean(test.acc)


