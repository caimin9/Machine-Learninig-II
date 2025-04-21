# --------------------------------------------------------
# ST4061 / ST6041 / ST6042
# 2024-25
# Eric Wolsztynski
# ...
# Exercises Section 6: Neural Networks
# --------------------------------------------------------

# when linear.output = F output layer uses same activation as input layer
# when T output layer uses a linear activation function
# T for Regression , F for classification

## rep = 5 -->
#Trains 5 separate neural networks with different random initializations
#Each network starts with different random weights
#The algorithm then selects the best performing network (lowest error)
#Helps avoid poor local minima that a single training run might get stuck in
#Improves model stability but increases training time
###############################################################
### Exercise 1
###############################################################

rm(list=ls())

library(MASS)
library(neuralnet)
# --- NN with one 10-node hidden layer
nms = names(Boston)[-14]
f = as.formula(paste("medv ~", paste(nms, collapse = " + ")))
set.seed(4061)
out.nn = neuralnet(f, data=Boston, hidden=c(10), rep=5, 
	linear.output=FALSE)
summary(out.nn$response)

# without using an activation function:
set.seed(4061)
out.nn.lin = neuralnet(f, data=Boston, hidden=c(10), rep=1, 
	linear.output=TRUE)
# Warning message:
# Algorithm did not converge in 1 of 1 repetition(s) within the stepmax. 

set.seed(4061)
out.nn.tanh = neuralnet(f, data=Boston, hidden=c(10), rep=5, 
	linear.output=FALSE, act.fct='tanh')

p1 = predict(out.nn, newdata=Boston)
p2 = predict(out.nn.tanh, newdata=Boston)
sqrt(mean((p1-Boston$medv)^2))
sqrt(mean((p2-Boston$medv)^2))

###############################################################
### Exercise 2
###############################################################

library(neuralnet) 
set.seed(4061)
n = nrow(iris)
dat = iris[sample(1:n), ] # shuffle initial dataset
NC = ncol(dat)
nno = neuralnet(Species~., data=dat, hidden=c(6,5))
plot(nno)

###############################################################
### Exercise 3
###############################################################

rm(list=ls())

library(neuralnet)
library(nnet)    # implements single layer NNs
library(MASS) # includes dataset BostonHousing

data(Boston) # load the dataset

# train neural nets
set.seed(4061)
n = nrow(Boston)
itrain = sample(1:n, round(.7*n), replace=FALSE)
dat = Boston
dat$medv = dat$medv/50
dat.train = dat[itrain,]
dat.test = dat[-itrain,-14]
y.test = dat[-itrain,14]

set.seed(4061)
nno1 = nnet(medv~., data=dat.train, size=5, linout=TRUE)
fit1 = nno1$fitted.values
mean((fit1-dat.train$medv)^2)

set.seed(4061)
nno2 = neuralnet(medv~., data=dat.train, hidden=c(5), linear.output=TRUE)
fit2 = predict(nno2, newdata=dat.train)[,1]
mean((fit2-dat.train$medv)^2)

set.seed(4061)
nms = names(dat)[-14]
f = as.formula(paste("medv ~", paste(nms, collapse = " + ")))
nno3 = neuralnet(f, data=dat.train, hidden=5, threshold=0.0001)
fit3 = predict(nno3, newdata=dat.train)[,1]
mean((fit3-dat.train$medv)^2)

# test neural nets
y.test = y.test*50
p1 = predict(nno1, newdata=dat.test)*50
p2 = predict(nno2, newdata=dat.test)*50
p3 = predict(nno3, newdata=dat.test)*50
mean((p1-y.test)^2)
mean((p2-y.test)^2)
mean((p3-y.test)^2)

# explain these differences??!!?!
names(nno1)
names(nno2)

# nnet:
# - activation function: logistic
# - algorithm: BFGS in optim
# - decay: 0
# - learning rate: NA
# - maxit: 100

# neuralnet:
# - activation function: logistic
# - algorithm: (some form of) backpropagation
# - decay: ?
# - learning rate: depending on algorithm
# - maxit:?

# so what is it?

###############################################################
### Exercise 4
###############################################################

rm(list=ls())

library(caret)
library(neuralnet)
library(nnet)
library(ISLR)

# set up the data (take a subset of the Hitters dataset)
dat = na.omit(Hitters) 
n = nrow(dat)
NC = ncol(dat)

# Then try again after normalizing the response variable to [0,1]:
dats = dat
dats$Salary = (dat$Salary-min(dat$Salary)) / diff(range(dat$Salary))

# train neural net
set.seed(4061)
itrain = sample(1:n, round(.7*n), replace=FALSE)
dat.train = dat[itrain,]
dats.train = dats[itrain,]
dat.test = dat[-itrain,]
dats.test = dats[-itrain,]

set.seed(4061)
nno = nnet(Salary~., data=dat.train, size=10, decay=c(0.1))
summary(nno$fitted.values)

set.seed(4061)
# 0 decay means no regularisation. As decay increases so does regularisation
nno.s = nnet(Salary~., data=dats.train, size=10, decay=c(0))
summary(nno.s$fitted.values)

set.seed(4061)
nno.s = nnet(Salary~., data=dats.train, size=10, decay=c(0.1))
summary(nno.s$fitted.values)

# Our last attempt above was a success.
# But we should be able to get a proper fit even for decay=0... 
# what's going on? Can you get it to work?

# (A1) Well, it's one of these small details in how you call a function;
# here we have to specify 'linout=1' because we are considering a 
# regression problem:

set.seed(4061)
nno = nnet(Salary~., data=dat.train, size=10, decay=c(0.1), linout=1)
summary(nno$fitted.values)

set.seed(4061)
nno.s = nnet(Salary~., data=dats.train, size=10, decay=c(0), linout=1)
summary(nno.s$fitted.values)

set.seed(4061)
nno.s = nnet(Salary~., data=dats.train, size=10, decay=c(0.1), linout=1)
summary(nno.s$fitted.values)

# (A2) but let's do the whole thing again more cleanly...

# re-encode and scale dataset properly
myrecode <- function(x){
# function recoding levels into numerical values
	if(is.factor(x)){
		levels(x)
		return(as.numeric(x)) 
	} else {
		return(x)
	}
}
myscale <- function(x){
# function applying normalization to [0,1] scale
	minx = min(x,na.rm=TRUE)
	maxx = max(x,na.rm=TRUE)
	return((x-minx)/(maxx-minx))
}
datss = data.frame(lapply(dat,myrecode))
datss = data.frame(lapply(datss,myscale))

# replicate same train-test split:
datss.train = datss[itrain,]
datss.test = datss[-itrain,]

set.seed(4061)
nno.ss.check = nnet(Salary~., data=datss.train, size=10, decay=0, linout=1)
summary(nno.ss.check$fitted.values)

# use same scaled data but with decay as before:
set.seed(4061)
nno.ss = nnet(Salary~., data=datss.train, size=10, decay=c(0.1), linout=1)
summary(nno.ss$fitted.values)

# evaluate on test data (with same decay for both models):
datss.test$Salary - dats.test$Salary
pred.s = predict(nno.s, newdata=dats.test)
pred.ss = predict(nno.ss, newdata=datss.test)
mean((dats.test$Salary-pred.s)^2)
mean((datss.test$Salary-pred.ss)^2)



###############################################################
### Exercise 5: neural networks using caret
###############################################################

rm(list=ls())

library(caret)
library(neuralnet)
library(nnet)
library(ISLR)
library(mlbench)

# Set up the data (same as Exercise 4)
dat = na.omit(Hitters) 
n = nrow(dat)
NC = ncol(dat)

# Prepare scaled data for better training
myrecode <- function(x) {
  if(is.factor(x)) {
    return(as.numeric(x)) 
  } else {
    return(x)
  }
}

myscale <- function(x) {
  minx = min(x, na.rm=TRUE)
  maxx = max(x, na.rm=TRUE)
  return((x-minx)/(maxx-minx))
}

datss = data.frame(lapply(dat, myrecode))
datss = data.frame(lapply(datss, myscale))

# Train-test split
set.seed(4061)
itrain = sample(1:n, round(.7*n), replace=FALSE)
datss.train = datss[itrain,]
datss.test = datss[-itrain,]

# 1. Fit a single-layer FFNN using caret with decay=0.1
set.seed(4061)
model1 <- train(
  Salary ~ ., 
  data = datss.train,
  method = "nnet",
  tuneGrid = data.frame(decay = 0.1, size = 10),
  linout = TRUE,
  trace = FALSE,
  maxit = 500
)

print(model1)
pred1 <- predict(model1, datss.test)
rmse1 <- sqrt(mean((pred1 - datss.test$Salary)^2))
cat("RMSE for fixed model:", rmse1, "\n")

# 2. Tune the single-layer FFNN and compare
set.seed(4061)
model2 <- train(
  Salary ~ .,                   
  data = datss.train,          
  method = "nnet",             
  tuneGrid = expand.grid(      # Grid of hyperparameters to try
    decay = c(0, 0.001, 0.01, 0.1, 0.5),  # Weight decay values (regularisation)
    size = c(5, 10, 15)        # Number of neurons in hidden layer
  ),
  linout = TRUE,               # Linear output activation (for regression tasks)
  trace = FALSE,               # Suppress training details/progress output
  maxit = 500                  # Maximum number of training iterations
)

print(model2)
plot(model2)
pred2 <- predict(model2, datss.test)
rmse2 <- sqrt(mean((pred2 - datss.test$Salary)^2))
cat("RMSE for tuned model:", rmse2, "\n")

# 3. Use 3-layer FFNN with neuralnet and caret
# neuralnet implementation
set.seed(4061)
nn3 <- neuralnet(
  Salary ~ ., 
  data = datss.train,
  hidden = c(10, 5),
  linear.output = TRUE
)

pred3_nn <- predict(nn3, datss.test)
rmse3_nn <- sqrt(mean((pred3_nn - datss.test$Salary)^2))
cat("RMSE for neuralnet 3-layer:", rmse3_nn, "\n")

# caret implementation with mlp
set.seed(4061)
model3 <- train(
  Salary ~ ., 
  data = datss.train,
  method = "mlpML",
  tuneGrid = data.frame(size = 10),
  trControl = trainControl(method = "cv", number = 5)
)

print(model3)
pred3 <- predict(model3, datss.test)
rmse3 <- sqrt(mean((pred3 - datss.test$Salary)^2))
cat("RMSE for mlpML:", rmse3, "\n")

# 4. Fine-tune with mlpML
set.seed(4061)
model4 <- train(
  Salary ~ ., 
  data = datss.train,
  method = "mlpML",
  tuneGrid = expand.grid(size = c(5, 10, 15, 20)),
  trControl = trainControl(method = "cv", number = 5)
)

print(model4)
plot(model4)
pred4 <- predict(model4, datss.test)
rmse4 <- sqrt(mean((pred4 - datss.test$Salary)^2))
cat("RMSE for tuned mlpML:", rmse4, "\n")

# Compare results
results <- data.frame(
  Model = c("nnet fixed", "nnet tuned", "neuralnet 3-layer", "mlpML", "mlpML tuned"),
  RMSE = c(rmse1, rmse2, rmse3_nn, rmse3, rmse4)
)
print(results)




###############################################################
### Exercise 6: Olden index
###############################################################

rm(list=ls())

library(nnet)
library(NeuralNetTools)
library(randomForest)
library(MASS)

myscale <- function(x){
	minx = min(x,na.rm=TRUE)
	maxx = max(x,na.rm=TRUE)
	return((x-minx)/(maxx-minx))
}

# (1) Iris data

# shuffle dataset...
set.seed(4061)
n = nrow(iris)
dat = iris[sample(1:n),]

# rescale predictors...
dat[,1:4] = myscale(dat[,1:4])

# fit Feed-Forward Neural Network...
set.seed(4061)
nno = nnet(Species~., data=dat, size=c(7), 
	linout=FALSE, entropy=TRUE)
pis = nno$fitted.values
matplot(pis, col=c(1,2,4), pch=20)
y.hat = apply(pis, 1, which.max) # fitted values
table(y.hat, dat$Species)

# compute variable importance...
vimp.setosa = olden(nno, out_var='setosa', bar_plot=FALSE)
vimp.virginica = olden(nno, out_var='virginica', bar_plot=FALSE)
vimp.versicolor = olden(nno, out_var='versicolor', bar_plot=FALSE)
names(vimp.setosa)
par(mfrow=c(1,2))
plot(iris[,3:4], pch=20, col=c(1,2,4)[iris$Species], cex=2)
plot(iris[,c(1,3)], pch=20, col=c(1,2,4)[iris$Species], cex=2)
dev.new()
plot(olden(nno, out_var='setosa'))
plot(olden(nno, out_var='virginica'))
plot(olden(nno, out_var='versicolor'))
v.imp = cbind(vimp.setosa$importance, vimp.virginica$importance, vimp.versicolor$importance)
rownames(v.imp) = names(dat)[1:4]
colnames(v.imp) = levels(dat$Species)
(v.imp)

# fit RF...
set.seed(4061)
rfo = randomForest(Species~., data=dat, ntrees=1000)
rfo$importance

# how can we compare variable importance assessments?
cbind(apply(v.imp, 1, sum), 
	apply(abs(v.imp), 1, sum), 
	rfo$importance)

# (2) Boston data

set.seed(4061)
n = nrow(Boston)
dat = Boston[sample(1:n),]

# rescale predictors...
dats = myscale(dat)
dats$medv = dat$medv/50

set.seed(4061)
nno = nnet(medv~., data=dats, size=7, linout=1)
y.hat = nno$fitted.values
plot(y.hat*50, dat$medv)
mean((y.hat*50-dat$medv)^2)

v.imp = olden(nno, bar_plot=FALSE)
plot(v.imp)

# fit RF...
set.seed(4061)
rfo = randomForest(medv~., data=dat, ntrees=1000)
rfo$importance

# how can we compare variable importance assessments?
cbind(v.imp, rfo$importance)
round(cbind(v.imp/sum(abs(v.imp)), 
		rfo$importance/sum(rfo$importance)),3)*100

# should we use absolute values of Olden's index?
par(mfrow=c(2,1))
barplot(abs(v.imp[,1]), main="importance from NN", 
	names=rownames(v.imp), las=2)
barplot(rfo$importance[,1], main="importance from RF", 
	names=rownames(v.imp), las=2)
	
# or possibly normalize across all values for proportional contribution?
par(mfrow=c(2,1))
NNN = sum(abs(v.imp[,1]))
NRF = sum(abs(rfo$importance[,1]))
barplot(abs(v.imp[,1])/NNN, main="importance from NN", 
	names=rownames(v.imp), las=2)
barplot(rfo$importance[,1]/NRF, main="importance from RF", 
	names=rownames(v.imp), las=2)

# looks alright... now make it a nicer comparative plot :)
par(font=2, font.axis=2)
imps = rbind(NN=abs(v.imp[,1])/NNN, RF=rfo$importance[,1]/NRF)
cols = c('cyan','pink')
barplot(imps, names=colnames(imps), las=2, beside=TRUE, 
	col=cols,
	ylab="relative importance (%)",
	main="Variable importance from NN and RF")
legend("topleft", legend=c('NN','RF'), col=cols, bty='n', pch=15)
