# --------------------------------------------------------
# ST4061 / ST6041 / ST6042
# 2024-25
# Eric Wolsztynski
# ...
# Section 6: demo code for neural networks
# In this script we illustrate basic implementation of 
# neural networks for classification and regression 
# settings.
# Note that all illustrations of performance in this code 
# are for illustration; in particular they should at least 
# be cross-validated.
# --------------------------------------------------------

rm(list=ls())  # clear out running environment

# ------------------------------------------------------------
# Example 1: iris data with neuralnet

library(neuralnet) 

set.seed(4061)
n = nrow(iris)
dat = iris[sample(1:n), ] # shuffle initial dataset
NC = ncol(dat)
nno = neuralnet(Species~., data=dat, hidden=c(6,5))
plot(nno, information=FALSE, col.entry='red', col.out='green',
	show.weights=FALSE)
plot(nno, information=TRUE, col.entry='red', col.out='green',
	show.weights=TRUE)

# ------------------------------------------------------------
# Example 2: single layer NN - regression - effect of scaling

rm(list=ls())    # clear running environment (memory)

library(nnet)    # implements single layer NNs
library(mlbench) # includes dataset BostonHousing

data(BostonHousing) # load the dataset

# train neural net
n = nrow(BostonHousing)
itrain = sample(1:n, round(.7*n), replace=FALSE)
nno = nnet(medv~., data=BostonHousing, subset=itrain, size=5)
summary(nno$fitted.values)
# the above output indicates the algorithm did not 
# converge, probably due to explosion of the gradients...
#
# We try again, this time normalizing the values
# (50 is the max value for this data, see its range):
head(BostonHousing)
summary(BostonHousing$medv)
nno = nnet(medv/50~., data=BostonHousing, subset=itrain, size=5)
summary(nno$fitted.values)
plot(BostonHousing$medv[itrain]/50,nno$fitted.values,pch=20)
abline(a=0,b=1,lwd=4, col ='red')
# there was thus a need to normalise the response variable...

# test neural net
preds = predict(nno, newdata=BostonHousing[-itrain,]) * 50
# (we multiply by 50 to fall back to original data domain)
summary(preds)
# RMSE:
sqrt(mean((preds-BostonHousing$medv[-itrain])^2))

# compare with lm():
lmo = lm(medv~., data=BostonHousing, subset=itrain)
lm.preds = predict(lmo, newdata= BostonHousing[-itrain,])
# RMSE:
sqrt(mean((lm.preds-BostonHousing$medv[-itrain])^2))

# and now applying all scalings correctly:
names(BostonHousing)
head(BostonHousing)
BH = BostonHousing[,-grep("medv",names(BostonHousing))]
minmax.scale <- function(x){
	return( (x-min(x))/max(x) )
}
fac = which(unlist(lapply(BH,class)) == "factor") #Finds the categorical variables
BH[,-fac] = apply(BH[,-fac],2,minmax.scale) #2 here just means we apply the minmax.scale fnc column wise (helps dealing with exploding gradients)
BH[,fac] = (as.numeric(BH[,fac])-1) #Turns them into binary 0 or 1, hence the -1
BH$medv = BostonHousing$medv/50
nno = nnet(medv~., data=BH, subset=itrain, size=5)
summary(nno$fitted.values)
preds = predict(nno, newdata=BH[-itrain,]) * 50
summary(preds)
# RMSE:
sqrt(mean((preds-BostonHousing$medv[-itrain])^2))
# recall lm RMSE:
sqrt(mean((lm.preds-BostonHousing$medv[-itrain])^2))

# Further diagnostics may highlight various aspects of the 
# model fit - always run these checks!
par(mfrow=c(2,2))
plot(lmo$residuals, nno$residuals*50)
abline(a=0, b=1, col="limegreen",lwd=3)
plot(BostonHousing$medv[itrain], lmo$residuals, pch=20)
plot(BostonHousing$medv[itrain], lmo$residuals, pch=20, col=8)
points(BostonHousing$medv[itrain], nno$residuals*50, pch=20)
qqnorm(nno$residuals)
abline(a=mean(nno$residuals), b=sd(nno$residuals), col=2)





"""
Key Differences
Aspect						Method 1				Method 2
Input features					Unscaled				All inputs scaled to [0,1]
Factor treatment				Handled by nnet internally		Explicitly converted to 0/1
Target scaling					Yes					Yes
Risk of exploding/vanishing gradients		Higher					Lower
Training convergence				May fail or converge poorly		More stable
RMSE (test)					Usually worse				Usually better


Why does this matter?
Neural networks — especially shallow ones like those in nnet — are sensitive to scale. If input features vary widely (e.g. rm ranges 3–8, tax ranges 200–700), then:
Gradients can explode or vanish.
Weight updates become erratic.
It becomes harder to converge on a good solution.

Scaling all features ensures that:
The model can learn smoother gradient steps.
Each feature contributes roughly equally during training.

Factor encoding also matters: nnet will internally one-hot encode factors, but if you convert them manually to numeric, you control exactly how they're used. 
In this case chas is binary anyway, so both approaches work — but it's safer and clearer to do it explicitly
"""

# ------------------------------------------------------------
# Example 3: effect of tuning parameters (iris data)

rm(list=ls())

set.seed(4061)
n = nrow(iris)
# shuffle initial dataset as per usual
# (removing 4th predictor to make it more tricky)
dat = iris[sample(1:n),-4] 
NC = ncol(dat)

# Normalize data to [0,1] using transformation
# y_normalized = (y-min(y)) / (max(y)-min(y)):
mins = apply(dat[,-NC],2,min)
maxs = apply(dat[,-NC],2,max)
dats = dat
dats[,-NC] = scale(dats[,-NC],center=mins,scale=maxs-mins)

# train neural net on training set:
itrain = sample(1:n, round(.7*n), replace=FALSE)
nno = nnet(Species~., data=dats, subset=itrain, size=5)

# generate predictions for test set:
nnop = predict(nno, dats[-itrain,])
head(nnop)
# that's one way of getting predicted labels from 
# probabilites:
preds = max.col(nnop) 
# (the above line picks the column with highest proba 
# for each row, i.e. each observation)
# alternatively we could use this directly:
preds = predict(nno, dats[-itrain,], type='class')
tbp = table(preds, dats$Species[-itrain])
sum(diag(tbp))/sum(tbp)

# effect of size on classification performance?
# Here we try sizes 1 to 10 for illustrative purposes
# but feel free to mess around with these values!
sizes = c(1:10)
rate = numeric(length(sizes)) # train-set classification rate
ratep = numeric(length(sizes)) # test-set classification rate
for(d in 1:length(sizes)){
	nno = nnet(Species~., data=dats, subset=itrain, 
		size=sizes[d])
	tb = table(max.col(nno$fitted.values), dats$Species[itrain])
	rate[d] = sum(diag(tb))/sum(tb)
	# now looking at test set predictions
	nnop = predict(nno, dats[-itrain,])
	tbp = table(max.col(nnop), dats$Species[-itrain])
	ratep[d] = sum(diag(tbp))/sum(tbp)
}
plot(rate, pch=20, t='b', xlab="layer size", ylim=range(c(rate,ratep)))
points(ratep, pch=15, t='b', col=2)
legend('bottomright', legend=c('training','testing'), 
		pch=c(20,15), col=c(1,2), bty='n')
# Notice how train- and test-set performances are not
# necessarily similar... 

# effect of decay?
decays = seq(1,.0001,lengt=11)
rate = numeric(length(decays)) # train-set classification rate
ratep = numeric(length(decays)) # test-set classification rate
for(d in 1:length(decays)){
	# fit NN with that particular decay value (decays[d]):
	nno = nnet(Species~., data=dats, subset=itrain, size=10, 
			decay=decays[d])
	# corresponding train set confusion matrix:
	tb = table(max.col(nno$fitted.values), dats$Species[itrain])
	rate[d] = sum(diag(tb))/sum(tb)
	# now looking at test set predictions:
	nnop = predict(nno, dats[-itrain,])
	tbp = table(max.col(nnop), dats$Species[-itrain])
	ratep[d] = sum(diag(tbp))/sum(tbp)
}
plot(decays, rate, pch=20, t='b', ylim=range(c(rate,ratep)))
points(decays, ratep, pch=15, t='b', col=2)
legend('topright', legend=c('training','testing'), 
	pch=c(20,15), col=c(1,2), bty='n')
