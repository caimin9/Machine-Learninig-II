# --------------------------------------------------------
# ST4061 / ST6041 / ST6042
# 2024-25
# Eric Wolsztynski
# ...
# Section 7: feature selection
# --------------------------------------------------------

rm(list=ls())

library(ISLR)
library(caret)
library(randomForest)

set.seed(6041)
dat1 = na.omit(Hitters)
x = dat1
x$Salary = NULL
y = log(dat1$Salary)
n = nrow(x)
p = ncol(x)

# ------------------------------------------
# effect of redundant information 
# (CoD)

set.seed(4061)
rfo1 = randomForest(x=x,y=y)
K = 10
res = numeric(K)
for(k in 1:K){
	x2 = cbind(x, matrix(rnorm(n*k*p),nrow=n))
	rfo2 = randomForest(x=x2,y=y)
	res[k] = mean(rfo2$mse)
}

par(mfrow=c(1,2))
resf = c(mean(rfo1$mse),res)
plot(c(0:K),resf,t='b',pch=20,
	main="OOB RMSE",xlab="k",ylab="")
resp = (resf-resf[1])/resf[1]
plot(c(1:K),resp[-1],t='b',pch=20,
	main="%-change in OOB RMSE",xlab="k",ylab="")

# ------------------------------------------
# selection bias

dat = read.csv('/Users/ewol/Downloads/sarcoma.csv',stringsAsFactors=TRUE)

# create binary data (3-yr survival)
i.rm = which((dat$surtim<36) & (dat$surind==0))
dat = dat[-i.rm,]
dat = na.omit(dat)
x = dat
x$surtim = NULL
x$surind = NULL
y = as.factor((dat$surtim<36) & (dat$surind==1))
levels(y) = c('No','Yes')
dim(dat)
n = nrow(x)
p = ncol(x)

# vanilla model
set.seed(1)
o1 = caret::train(x,y)
o1

# look for a better model?

my.filter <- function(x,y,co=.05){
	p = ncol(x)
	pvals = numeric(p)
	for(i in 1:p){
		pvals[i] = wilcox.test(as.numeric(x[,i])~y)$p.value
	}
	i.sel = which(pvals<co)
	return(x[,i.sel])
}

xf = my.filter(x,y,co=.2)
set.seed(1)
o2 = caret::train(xf,y)
o2

# doing it manually... 
B = 25
accf = acc = accf2 = numeric(B)
set.seed(3)
for(b in 1:B){
	# create bootstrap data
	ib = sample(1:n,n,replace=TRUE)
	xb = x[ib,]
	yb = y[ib]
	xt = x[-unique(ib),]
	yt = y[-unique(ib)]
	#
	# train model with no filter
	rfb = randomForest(x=xb,y=yb)
	yp = predict(rfb,xt)
	tb = table(yp,yt)
	acc[b] = sum(diag(tb))/sum(tb)
	#
	# filter (the correct way!)
	xbf = my.filter(xb,y,co=.2)
	rfbf = randomForest(x=xbf,y=yb)
	yp = predict(rfbf,xt)
	tb = table(yp,yt)
	accf[b] = sum(diag(tb))/sum(tb)
	#
	# from overall filter?
	xb = xf[ib,]
	rfbf2 = randomForest(x=xb,y=yb)
	yp = predict(rfbf2,xt)
	tb = table(yp,yt)
	accf2[b] = sum(diag(tb))/sum(tb)
}
mean(acc)
mean(accf)
mean(accf2)
boxplot(acc,accf,accf2)



o = glm(y~.,data=x,family='binomial')
summary(o)