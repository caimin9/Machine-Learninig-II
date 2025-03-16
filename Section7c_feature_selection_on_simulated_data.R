# --------------------------------------------------------
# ST4061 / ST6041 / ST6042
# 2024-25
# Eric Wolsztynski
# ...
# Demo Section 7: model selection
# * Forward/Backward subset selection
# * RFE from caret (Recursive Feature Elimination)
# --------------------------------------------------------

#########################################################  
### stepwise (logistic) regression with caret
######################################################### 

rm(list=ls())
set.seed(4061)

dat = na.omit(Hitters)
n = nrow(dat)
dat = dat[sample(1:n,n),]
dat$y = as.factor(dat$Salary>mean(dat$Salary))
dat$Salary = NULL
levels(dat$y) = c("low","hi")

# Stratified split into train + test datasets
set.seed(4060)
itrain = createDataPartition(dat$y, p=.75, times=1)[[1]]
dtrain = dat[itrain,]
dtest = dat[-itrain,] 

trC = trainControl(method="cv", number=5,
                                  savePredictions = TRUE, 
                                  classProbs = TRUE)
co = train(y~., data=dtrain, method='glmStepAIC', 
			trControl=trC, distribution='binomial') 
summary(co$finalModel)
names(co)
preds <- predict(co, dtest) 
probs <- predict(co, dtest, type="prob") 
table(dtest$y,preds)
pd = data.frame(obs=dtest$y,pred=preds,low=probs$low)
twoClassSummary(pd, lev=levels(dtest$y))

#########################################################  
### on simulated data
######################################################### 

rm(list=ls())

library(glmnet)
library(leaps)
library(caret)

set.seed(4061)

# We start by creating a dummy data set, so that we know which 
# features actually make up the observations. Here a linear
# combination of nonlinear expression of 3 features are used
# to create observations Y. 
# The 3 features are:
# - education level 
# - number of years of experience
# - some employee rating assessed by their company
# The response variable Y is salary.
# Another 20 features containing random noise are also created
# and added to the dataset. They should not be selected by our
# model selection method...

n = 500 # desired sample size

# level of education (1=secondary level, 2=BSc, 3=MSc, 4=PhD/MBA):
edu = sample(c(1:4), size=n, prob=c(.05,.5,.35,.1), replace=TRUE)
# nbr years experience:
yex = rpois(n=n, lambda=6) 
# some obscure employee rating made up by the company:
ert = pmin(5, pmax(0, 5-rexp(n=n, rate=2)))
# employee salary (response variable):
sal = 2*exp(.15*yex) + 3.2*log(edu) + 4*ert

par(mfrow=c(2,2))
plot(factor(edu), main="education")
hist(yex, main="years experience")
hist(ert, main="employee rating")
hist(sal, main="Salaries")

par(mfrow=c(1,3), pch=20)
boxplot(sal~factor(edu), main="salary wrt\n education")
plot(yex, sal, main="salary v\n years experience")
plot(ert, sal, main="salary v\n employee rating")

# now make up some dummy features...
# we don't bother changes scales/means since we will
# be normalizing these features...
p = 20
xtra = matrix(rnorm(n*p), ncol=p, nrow=n)
colnames(xtra) = paste("X",c(1:p),sep="")

par(mfrow=c(4,5), pch=20, mar=c(1,1,1,1))
for(j in 1:p){
	plot(xtra[,j], sal, main=paste(j))
}

# the data frame(s):
features = data.frame(edu,yex,ert,xtra)
dat = data.frame(sal,features) # may be more convenient sometimes

# train-test split:
i.train = sample(1:n, size=300, replace=FALSE)

# line up data in several formats for convenience:
dat.train = dat[i.train,]
dat.test = dat[-i.train,]
x.train = features[i.train,] 
y.train = sal[i.train]
x.test = features[-i.train,] 
y.test = sal[-i.train]
# not forgetting matrix forms for the likes of glmnet:
xm = model.matrix(sal~.,data=features)[,-1]
xm.train = xm[i.train,]
xm.test = xm[-i.train,]

#########################################################  
### check out what LASSO would tell us
######################################################### 

lasso.cv = cv.glmnet(xm.train, y.train)
lasso = glmnet(xm.train, y.train, lambda=lasso.cv$lambda.min)
coef(lasso)
# c.lasso = caret::train(x.train, y.train, method="glmnet")

# how about cross-validating this?
set.seed(4061) # for reproducibility
K = 10
folds = cut(1:n, breaks=K, labels=FALSE)
sel = matrix(0, nrow=K, ncol=ncol(features))
colnames(sel) = names(features)
for(k in 1:K){
	itr = which(folds!=k)
	lasso.cv = cv.glmnet(xm[itr,], sal[itr])
	lasso = glmnet(xm[itr,], sal[itr], lambda=lasso.cv$lambda.min)
	isel = which(coef(lasso)[-1] != 0)
	sel[k,isel] = 1
}
apply(sel,2,mean)*100
# LASSO thinks X1 and X14, for example, are important...
# We'd be better off increasing the regularization parameter, 
# e.g. using lasso.cv$lambda.min*2 instead (try it!).

#########################################################  
### perform FS with caret::rfe based on linear regression 
######################################################### 

set.seed(4061)
subsets <- c(1:5, 10, 15, 20, ncol(features))
ctrl <- rfeControl(functions = lmFuncs,
                   method = "cv",
                   number = 10,
                   # method = "repeatedcv",
                   # repeats = 5,
                   verbose = FALSE)
lm.rfe <- rfe(x.train, y.train,
                 sizes = subsets,
                 rfeControl = ctrl)

lm.rfe 
# This function has picked the correct subset of features

#########################################################  
### compare with leaps...
######################################################### 

set.seed(4061)
reg.bwd = regsubsets(sal~., data=dat.train, 	
	nvmax=ncol(features))
opt.mod = which.max(summary(reg.bwd)$adjr2)
isel = which(summary(reg.bwd)$which[opt.mod,-1])
isel

# how about cross-validating this?
set.seed(4061) # for reproducibility
K = 10
folds = cut(1:n, breaks=K, labels=FALSE)
sel = matrix(0, nrow=K, ncol=ncol(features))
colnames(sel) = names(features)
for(k in 1:K){
	itr = which(folds!=k)
	reg.bwd = regsubsets(sal~., data=dat[itr,], 
				nvmax=ncol(features))
	opt.mod = which.max(summary(reg.bwd)$adjr2)
	isel = which(summary(reg.bwd)$which[opt.mod,-1])
	sel[k,isel] = 1
}
apply(sel,2,mean)*100
# X1 and X14, again...

#########################################################  
### perform FS with caret::rfe based on RF
######################################################### 

set.seed(4061)
subsets <- c(1:5, 10, 15, 20, ncol(features))
ctrl <- rfeControl(functions = rfFuncs,
                   method = "cv",
                   number = 10,
                   # method = "repeatedcv",
                   # repeats = 5,
                   verbose = FALSE)
rf.rfe <- rfe(x.train, y.train,
                 sizes = subsets,
                 rfeControl = ctrl)
rf.rfe
# worth the wait!
