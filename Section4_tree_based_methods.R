# --------------------------------------------------------
# ST4061 / ST6041 / ST6042
# 2024-25
# Eric Wolsztynski
# ...
# Exercises Section 4: Tree-based methods
# --------------------------------------------------------

###############################################################
### Exercise 1: growing and pruning a tree
###############################################################

library(ISLR) # contains the dataset
library(tree) # contains... tree-building methods

# Recode response variable so as to make it a classification problem
High = ifelse(Carseats$Sales<=8, "No", "Yes")
# Create a data frame that includes both the predictors and response
# (a data frame is like an Excel spreadsheet, if you like)
CS = data.frame(Carseats, High)
CS$Sales = NULL
CS$High = as.factor(CS$High) # <-- this bit was missing

# Fit the tree using all predictors (except for variable Sales, 
# which we have "recoded" into a cateorical response variable)
# to response variable High
tree.out = tree(High~., CS)
summary(tree.out)
# plot the tree
plot(tree.out)
text(tree.out, pretty=0)

# pruning:
set.seed(3)
cv.CS = cv.tree(tree.out, FUN=prune.misclass)
names(cv.CS)
# - size:
# number of terminal nodes in each tree in the cost-complexity pruning sequence.
# - deviance:	
# total deviance of each tree in the cost-complexity pruning sequence.
# - k:
# the value of the cost-complexity pruning parameter of each tree in the sequence.
cv.CS
par(mfrow=c(1,2))
plot(cv.CS$size,cv.CS$dev,t='b')
abline(v=cv.CS$size[which.min(cv.CS$dev)])
plot(cv.CS$k,cv.CS$dev,t='b')

# use pruning: 
# - use which.min(cv.CS$dev) to get the location of the optimum
# - retrieve the corresponding tree size
# - pass this information on to pruning function
opt.size = cv.CS$size[which.min(cv.CS$dev)]
# see:
	plot(cv.CS$size,cv.CS$dev,t='b')
	abline(v=cv.CS$size[which.min(cv.CS$dev)])
ptree = prune.misclass(tree.out, best=opt.size)
ptree 
summary(ptree)
par(mfrow=c(1,2))
plot(tree.out)
text(tree.out, pretty=0)
plot(ptree)
text(ptree, pretty=0)


############################################################### 
### Exercise 2: apply CV and ROC analysis
############################################################### 

# Train/test:
set.seed(2)
n = nrow(CS)
itrain = sample(1:n, 200)
CS.test = CS[-itrain,]
High.test = High[-itrain]
# argument 'subset' makes it easy to handle training/test splits:
tree.out = tree(High~., CS, subset=itrain)
summary(tree.out)
plot(tree.out)
text(tree.out, pretty=0)

# fit error from full tree:
CS.train = CS[itrain,]
High.train = High[itrain]
tree.fit = predict(tree.out, CS.train, type="class")
ptree.fit = predict(ptree, CS.train, type="class")
(tb1 = table(tree.fit,High.train))
(tb2 = table(ptree.fit,High.train))
sum(diag(tb1))/sum(tb1)
sum(diag(tb2))/sum(tb2)
# fit is better from fully grown tree

# prediction from full tree:
tree.pred = predict(tree.out, CS.test, type="class")
(tb1 = table(tree.pred,High.test))
# prediction from pruned tree:
ptree.pred = predict(ptree, CS.test, type="class")
(tb2 = table(ptree.pred,High.test)) # confusion matrix
sum(diag(tb1))/sum(tb1)
sum(diag(tb2))/sum(tb2)

# perform ROC analysis
library(pROC)
# here we specify 'type="vector"' to retrieve continuous scores
# as opposed to predicted labels, so that we can apply varying
# threshold values to these scores to build the ROC curve:
ftree.probs = predict(tree.out, CS.test, type="vector")
roc.f = roc(response=(High.test), predictor=ftree.probs[,1])
roc.f$auc
ptree.probs = predict(ptree, CS.test, type="vector")
roc.p = roc(response=(High.test), predictor=ptree.probs[,1])
roc.p$auc
plot(roc.p)
plot(roc.f,add=TRUE,col=4)

############################################################### 
### Exercise 3: find the tree
############################################################### 

# ... can you find it?


############################################################### 
### Exercise 4: grow a random forest
############################################################### 

library(tree)
library(ISLR)
library(randomForest)

# ?Carseats
High = as.factor(ifelse(Carseats$Sales <= 8, 'No', 'Yes'))
CS = data.frame(Carseats, High)
CS$Sales = NULL
P = ncol(CS)-1  # number of features

# grow a single (unpruned) tree
tree.out = tree(High~., CS)

# fitted values for "training set"
tree.yhat = predict(tree.out, CS, type="class")

# grow a forest:
rf.out = randomForest(High~., CS)
# fitted values for "training set"
rf.yhat = predict(rf.out, CS, type="class")

# compare to bagging:
bag.out = randomForest(High~., CS, mtry=P)
# fitted values for "training set"
bag.yhat = predict(bag.out, CS, type="class")

# confusion matrix for tree:
(tb.tree = table(tree.yhat, High))
# confusion matrix for RF 
(tb.rf = table(rf.yhat, High))
# Note this is different to the confusion
# matrix for the OOB observations:
(tb.rf2 = rf.out$confusion)
# confusion matrix for bagging
(tb.bag = table(bag.yhat, High))

sum(diag(tb.tree))/sum(tb.tree)
sum(diag(tb.rf))/sum(tb.rf)
sum(diag(tb.bag))/sum(tb.bag)
sum(diag(tb.rf2))/sum(tb.rf2)

# train-test split
set.seed(6041)
N = nrow(CS)
itrain = sample(1:N, 200)
CS.train = CS[itrain,] 
CS.test = CS[-itrain,] 

tree.out = tree(High~., CS.train)
# fitted values for "train set"
tree.yhat = predict(tree.out, CS.train, type="class")
# fitted values for "test set"
tree.pred = predict(tree.out, CS.test, type="class")

rf.out = randomForest(High~., CS.train)
# fitted values for "training set"
rf.yhat = predict(rf.out, CS.train, type="class")
# fitted values for "test set"
rf.pred = predict(rf.out, CS.test, type="class")

bag.out = randomForest(High~., CS.train, mtry=(ncol(CS)-2))
# fitted values for "training set"
bag.yhat = predict(bag.out, CS.train, type="class")
# fitted values for "test set"
bag.pred = predict(bag.out, CS.test, type="class")

# confusion matrix for tree (test data):
(tb.tree = table(tree.pred, CS.test$High))
# confusion matrix for RF (test data):
(tb.rf = table(rf.pred, CS.test$High))
# confusion matrix for Bagging (test data):
(tb.bag = table(bag.pred, CS.test$High))

sum(diag(tb.tree))/sum(tb.tree)
sum(diag(tb.rf))/sum(tb.rf)
sum(diag(tb.bag))/sum(tb.bag)

############################################################### 
### Exercise 5: benchmarking (this exercise is left as homework)
############################################################### 

# bring in that code from Section 2 (below) and add to it:

library(class) # contains knn()
library(ISLR) # contains the datasets
library(pROC) 
library(tree)
library(randomForest)

## (1) benchmarking on unscaled data

set.seed(4061)
n = nrow(Default)
dat = Default[sample(1:n, n, replace=FALSE), ]

i.cv = sample(1:n, round(.7*n), replace=FALSE)
dat.cv = dat[i.cv,] # use this for CV (train+test)
dat.valid = dat[-i.cv,] # save this for later (after CV)

# tuning of the classifiers:
K.knn = 3 

K = 10
N = length(i.cv)
folds = cut(1:N, K, labels=FALSE)
acc.knn = acc.glm = acc.lda = acc.qda = numeric(K)
auc.knn = auc.glm = auc.lda = auc.qda = numeric(K)
acc.rf = auc.rf = numeric(K) 
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
	rf.o = randomForest(default~., data=dat.train)
	# test classifiers:
	# (notice that predict.glm() does not have a functionality to
	# return categorical values, so we copmute them based on the 
	# scores by applying a threshold of 50%)
	knn.p = knn.o
	glm.p = ( predict(glm.o, newdata=dat.test, type="response") > 0.5 )
	lda.p = predict(lda.o, newdata=dat.test)$class
	qda.p = predict(qda.o, newdata=dat.test)$class	
	rf.p = predict(rf.o, newdata=dat.test)
	# corresponding confusion matrices:
	tb.knn = table(knn.p, y.test)
	tb.glm = table(glm.p, y.test)
	tb.lda = table(lda.p, y.test)
	tb.qda = table(qda.p, y.test)
	tb.rf = table(rf.p, y.test)
	# store prediction accuracies:
	acc.knn[k] = sum(diag(tb.knn)) / sum(tb.knn)
	acc.glm[k] = sum(diag(tb.glm)) / sum(tb.glm)
	acc.lda[k] = sum(diag(tb.lda)) / sum(tb.lda)
	acc.qda[k] = sum(diag(tb.qda)) / sum(tb.qda)
	acc.rf[k] = sum(diag(tb.rf)) / sum(tb.rf)
	#
	# ROC/AUC analysis:
	# WARNING: THIS IS NOT PR(Y=1 | X), BUT Pr(Y = Y_hat | X):
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
boxplot(auc.knn, auc.glm, auc.lda, auc.qda,
	main="Overall CV AUC",
	names=c("kNN","GLM","LDA","QDA"))

###############################################################
### Exercise 6: Variable importance from RF
###############################################################

library(ISLR)
library(randomForest)

# ?Carseats
High = as.factor(ifelse(Carseats$Sales <= 8, 'No', 'Yes'))
CS = data.frame(Carseats, High)
CS$Sales = NULL

# grow a forest:
rf.out = randomForest(High~., CS)
# compare to bagging:
bag.out = randomForest(High~., CS, mtry=(ncol(CS)-1))

cbind(rf.out$importance, bag.out$importance)

par(mfrow=c(1,2))
varImpPlot(rf.out, pch=15, main="Ensemble method 1")
varImpPlot(bag.out, pch=15, main="Ensemble method 2")

###############################################################
### Exercise 7: gradient boosting 
###############################################################

library(ISLR) # contains the dataset
library(tree) # contains... tree-building methods
library(gbm)  # contains the GB model implementation
library(pROC)

# Recode response variable so as to make it a classification problem
High = as.factor(ifelse(Carseats$Sales<=8, "No", "Yes"))
CS = data.frame(Carseats, High)
# remove Sales from data frame, just to make formulas simpler!
CS$Sales = NULL  

set.seed(1)
itest = sample(1:nrow(CS), round(nrow(CS)/4))
CS.test = CS[itest,]
CS = CS[-itest,]

set.seed(4061)
# Note:
gbm(High~., data=CS, distribution="bernoulli")
# so we must recode the levels...
CS$High = (as.numeric(CS$High=="Yes")) # this hack could be useful 
gb.out = gbm(High~., data=CS, 
	distribution="bernoulli", # use "gaussian" instead for regression
	n.trees=5000, # size of the ensemble
	interaction.depth=1) # depth of the trees, 1 = stumps
summary(gb.out$train.error)
# inspect output:
par(mar=c(4.5,6,1,1))
summary(gb.out, las=1)
plot(gb.out, i="Price")
plot(gb.out, i="ShelveLoc")

gb.p = predict(gb.out, newdata=CS.test, n.trees=5000)
gb.p
roc.gb = roc(response=CS.test$High, predictor=gb.p)
plot(roc.gb)
roc.gb$auc

# compare AUC's with a Random Forest:
library(randomForest)
CS$High = as.factor(CS$High)
rf.out = randomForest(High~., CS, ntree=5000)
# fitted values for "training set"
rf.p = predict(rf.out, CS.test, type="prob")[,2] 
roc.rf = roc(response=CS.test$High, predictor=rf.p)
plot(roc.rf, add=TRUE, col=2)
roc.gb$auc
roc.rf$auc

###############################################################
### Exercise 8: gradient boosting using caret...
###############################################################

# Plug in the following snips of code within demo code
# Section4b_demo_using_caret.R:

############ (I) Classification example ############
### Gradient boosting (using caret) for classification

rm(list=ls()) # clear the environment
library(ISLR) # contains the data
library(caret) # contains everything else
set.seed(4061) # for reproducibility

# Set up the data (take a subset of the Hitters dataset)
data(Hitters)
Hitters = na.omit(Hitters)
dat = Hitters
n = nrow(dat)
NC = ncol(dat)
# Change the response variable to a factor to make this a 
# classification problem:
dat$Salary = as.factor(ifelse(Hitters$Salary>median(Hitters$Salary),
						"High","Low"))

# Data partition
itrain = sample(1:n, size=round(.7*n))
dat.train = dat[itrain,]
dat.validation = dat[-itrain,] # independent validation set for later
# x = select(dat.train,-"Salary") ### if using dplyr
# training set:
x = dat.train
x$Salary = NULL
y = as.factor(dat.train$Salary)

gb.out = train(Salary~., data=dat.train, method='gbm', distribution='bernoulli')
gb.fitted = predict(gb.out) # corresponding fitted values
gb.pred = predict(gb.out, dat.validation)
confusionMatrix(reference=dat.validation$Salary, data=gb.pred, 
							mode="everything")

############ (II) Regression example ############
### Gradient boosting (using caret) for regression

rm(list=ls()) # clear the environment

# Set up the data (take a subset of the Hitters dataset)
data(Hitters)
Hitters = na.omit(Hitters)
dat = Hitters
# hist(dat$Salary)
dat$Salary = log(dat$Salary)
n = nrow(dat)
NC = ncol(dat)

# Data partition
itrain = sample(1:n, size=round(.7*n))
dat.train = dat[itrain,]
dat.validation = dat[-itrain,]
x = dat.train
x$Salary = NULL
y = dat.train$Salary
ytrue = dat.validation$Salary

gb.out = train(Salary~., data=dat.train, method='gbm', distribution='gaussian')
gb.fitted = predict(gb.out) # corresponding fitted values
gb.pred = predict(gb.out, dat.validation)
mean((gb.pred-ytrue)^2)

