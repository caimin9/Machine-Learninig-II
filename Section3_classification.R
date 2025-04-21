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

set.seed(1)  # Set random seed for reproducible results
# Randomly shuffle the dataset to avoid any ordering bias
z = iris[sample(1:nrow(iris)),] 

# Visualize only sepal length and width (first 2 columns)
# col=c(1,2,4)[z[,5]] - colors points by species (black, red, blue)
# pch=20 - uses filled circle as point symbol
# cex=2 - increases point size by 2x
plot(z[,1:2], col=c(1,2,4)[z[,5]], 
	pch=20, cex=2)
x = z[,1:2]  # Extract only sepal measurements as predictors
y = z$Species  # Species column as the response variable

# Implementing k-Nearest Neighbors classifier
K = 5  # K parameter - number of neighbors to consider for classification
n = nrow(x)  # Number of observations
# Create train-test split (100 for training, 50 for testing)
i.train = sample(1:n, 100)  # Randomly select 100 indices for training
x.train = x[i.train,]  # Training predictors
x.test = x[-i.train,]  # Test predictors
y.train = y[i.train]  # Training response
y.test = y[-i.train]  # Test response
ko = knn(x.train, x.test, y.train, K)  # Run kNN algorithm
# knn() returns predicted classes for test data
tb = table(ko, y.test)  # Create confusion matrix
tb  # Display confusion matrix
1 - sum(diag(tb)) / sum(tb)  
# Calculate error rate: 1 - accuracy
# sum(diag(tb)) counts correct classifications (diagonal of confusion matrix)
# sum(tb) counts total observations

# More detailed assessment using confusionMatrix() from caret package
confusionMatrix(data=ko, reference=y.test)  # Shows accuracy, kappa, sensitivity, etc.

# Find optimal k value by testing different k values
# Maintaining the same train-test split for fair comparison
Kmax = 30  # Maximum k to try
acc = numeric(Kmax)  # Initialize accuracy vector to store results
for(k in 1:Kmax){
	ko = knn(x.train, x.test, y.train, k)  # Run kNN with current k
	tb = table(ko, y.test)  # Create confusion matrix
	acc[k] = sum(diag(tb)) / sum(tb)  # Calculate accuracy	
}
plot(1-acc, pch=20, t='b', xlab='k')  # Plot error rate vs k value
# pch=20 - solid circle, t='b' - both points and lines

# ROC analysis with kNN - Warning about limitations
# The knn() function doesn't provide class probabilities in the way we might expect
# For ROC analysis, we need Pr(Y=class | X), but knn gives different information

# Warning about knn's probability output
# When prob=TRUE, knn() returns the proportion of votes for the winning class
# Not the probability for each class, which limits ROC analysis
ko = knn(x.train, x.test, y.train, k)  # Basic call without probabilities
ko = knn(x.train, x.test, y.train, k, prob=TRUE)  # Request probability information
kp = attributes(ko)$prob  # Extract probability values
kp  # Display probabilities
# These are NOT probabilities that observation belongs to a specific class
# But rather the proportion of the k neighbors that voted for the assigned class
# This makes standard ROC analysis difficult, especially with 3+ classes


############################################################################################################################## 
### Exercise 2: GLM on 2-class iris data
##############################################################################################################################

n = nrow(iris)
is = sample(1:n, size=n, replace=FALSE)
dat = iris[is,-c(3,4)] # Using only sepal measurements, removing petal measurements

# Converting multi-class problem to binary classification
# Creating new binary target: 1 if virginica, 0 otherwise (setosa or versicolor)
dat$is.virginica = as.numeric(dat$Species=="virginica") 
dat$Species = NULL 
names(dat)

is = 1:100 # First 100 indices for training

# Fitting logistic regression model (GLM with binomial family and logit link)
# Logit link = log(p/(1-p)) where p is probability of is.virginica=1
fit = glm(is.virginica~., data=dat, subset=is,
				family=binomial(logit))

# Generate predictions on test set (51-150)
# type="response" returns probabilities instead of log-odds
pred = predict(fit, newdata=dat[-is,], type="response")
y.test = dat$is.virginica[-is] 

# Visualizing prediction distribution by true class
# Shows how well the model separates the classes
boxplot(pred~y.test, names=c("other","virginica"))
abline(h=0.5, col=3) # Standard decision threshold (0.5)
abline(h=0.1, col=4) # Alternative lower threshold - increases sensitivity

# Converting probabilities to binary predictions using 0.5 threshold
preds = as.factor(as.numeric(pred>.5))

# Creating and analyzing confusion matrix
# Shows counts of true positives, false positives, etc.
confusionMatrix(data=preds, reference=as.factor(y.test))

# Explicitly setting which class is considered "positive"
# Important for metrics like sensitivity and specificity
o = confusionMatrix(data=preds, reference=as.factor(y.test),
	positive="1")
o

# Extracting sensitivity from confusion matrix results
# Sensitivity = TP/(TP+FN) = proportion of actual positives correctly identified
names(o)
o$byClass[1]

# Threshold analysis - effect of different cutoff values on sensitivity
# Higher thresholds increase precision but reduce recall (sensitivity)
err = NULL
for(cut.off in seq(.1, .9, by=.1)){
	pred.y = as.numeric(pred>cut.off)
	
	# Creating confusion matrix with current threshold
	o = confusionMatrix(data=as.factor(pred.y), 
					reference=as.factor(y.test),
	positive="1")
	
	# Storing sensitivity for this threshold
	err = c(err,o$byClass[1])
}

# Plotting sensitivity vs threshold values
# Shows tradeoff in detection rate as threshold changes
plot(seq(.1, .9, by=.1), err, t='b')
##############################################################################################################################
### Exercise 3: LDA assumptions 
##############################################################################################################################
## (1) 2-class classification problem

dat = iris
dat$Species = as.factor(ifelse(iris$Species=="virginica",1,0))
# Recoding to create a binary classification problem (virginica vs others)

par(mfrow=c(1,2))
plot(iris[,1:2], pch=20, col=c(1,2,4)[iris$Species], cex=2)
legend("topright",col=c(1,2,4),
	legend=levels(iris$Species),
	pch=20, bty='n')
plot(dat[,1:2], pch=20, col=c(1,4)[dat$Species], cex=2)
legend("topright",col=c(1,4),
	legend=levels(dat$Species),
	pch=20, bty='n')
# Comparing original 3-class problem vs. binary reclassification

## Explore distribution of predictors by class:
par(mfrow=c(2,2))
for(j in 1:4){ 
	boxplot(dat[,j]~dat$Species, 
	      ylab='predictor',
				col=c('cyan','pink'), 
				main=names(dat)[j])
}
# Boxplots show distribution of each feature by class

# Checking for normality with histograms
par(mfrow=c(2,4), font.lab=2, cex=1.2)
for(j in 1:4){ 
	hist(dat[which(dat$Species=='other'),j], col='cyan', 
	      xlab='predictor for class other',
				main=names(dat)[j])
	hist(dat[which(dat$Species!='other'),j], col='pink', 
	     xlab='predictor for class virginica',
	     main=names(dat)[j])
}

# DataExplorer provides more sophisticated visualization options
library(DataExplorer)
plot_histogram(dat[,1:4])
plot_histogram(dat[which(dat$Species=='other'),1:4])
plot_histogram(dat[which(dat$Species=='virginica'),1:4])
plot_boxplot(dat,by='Species')

# QQ-plots provide better assessment of normality
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
# QQ-plots show if data follows normal distribution - points following line indicate normality

## Testing LDA assumptions statistically:
# 1. Equal variances across groups (homoscedasticity)
for(j in 1:4){
	print( bartlett.test(dat[,j]~dat$Species)$p.value )
}
# Bartlett's test - p > 0.05 indicates equal variances (good for LDA)

# 2. Normality within each group
for(j in 1:4){
  print( shapiro.test(dat[which(dat$Species=='virginica'),j])$p.value ) 
}
for(j in 1:4){
  print( shapiro.test(dat[which(dat$Species=='other'),j])$p.value ) 
}
# Shapiro-Wilk test - p > 0.05 indicates normality (good for LDA)

## Fit LDA model to this dataset and check accuracy:
lda.o = lda(Species~., data=dat)
(lda.o)
# LDA model output shows prior probabilities and group means

# Verification of summary values
table(dat$Species)/nrow(dat)  # Class prior probabilities
rbind(
    apply(dat[which(dat$Species=='other'),1:4], 2, mean),
    apply(dat[which(dat$Species=='virginica'),1:4], 2, mean)
)  # Group means

# Visualizing discriminant projections
x = as.matrix(dat[,1:4])
proj = x %*% lda.o$scaling  # Project data onto discriminant
plot(proj, pch=20, col=dat$Species, cex=2)

# Getting predictions and posterior probabilities
predo = predict(lda.o, newdata=dat)
y = predo$x
plot(proj, y)  # Should be identical
plot(y, predo$posterior[,2])  # Relationship between projection and posterior probability
boxplot(y ~ (predo$posterior[,2]>.5))
boxplot(proj ~ (predo$posterior[,2]>.5))

# Evaluating classification performance
fitted.values = predict(lda.o, newdata=dat)$class  
boxplot(y~dat$Species)
boxplot(proj~dat$Species)
(tb.2 = table(fitted.values, dat$Species))  # Confusion matrix
sum(diag(tb.2)) / sum(tb.2)  # Accuracy

## (2) 3-class classification problem

dat = iris
	
## Explore distribution of predictors:
par(mfrow=c(2,2))
for(j in 1:4){ 
	boxplot(dat[,j]~dat$Species,
				xlab = 'Species',
				ylab = 'predictor',
				col=c('cyan','pink'), 
				main=names(dat)[j])
}

# Checking normality for each species separately
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

# QQ-plots for more precise normality assessment
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

## Testing equal variances assumption for multi-class case
for(j in 1:4){
	print( bartlett.test(dat[,j]~dat$Species)$p.value )
}

## Fit LDA model to the multi-class dataset
lda.o = lda(Species~., data=dat)
(lda.o)
ftted.values = predict(lda.o, newdata=dat)$class
(tb.3 = table(ftted.values, dat$Species))  # Confusion matrix
sum(diag(tb.3)) / sum(tb.3)  # Accuracy

# Model Comparison:
# The 3-class LDA generally performs better than the binary LDA
# because:
# 1. The original classes are genuinely separate groups with distinct characteristics
# 2. Combining setosa and versicolor into one class ("other") mixes two distinct populations
# 3. By maintaining the natural class structure, the 3-class model can find better 
#    discriminant functions that separate all classes
# 4. The binary model forces LDA to find boundaries in a less natural grouping
# 5. Petal measurements are especially good at separating the three classes
##############################################################################################################################
### Exercise 4: LDA 
##############################################################################################################################
## (1) 2-class classification problem

dat = iris
dat$Species = as.factor(ifelse(iris$Species=="virginica",1,0))  # Create binary classification task
levels(dat$Species) = c("other","virginica")  # Rename factor levels for clarity

n = nrow(dat)
set.seed(4061)  # Set seed for reproducibility
dat = dat[sample(1:n),]  # Randomly shuffle dataset

i.train = 1:100  # First 100 samples for training
dat.train = dat[i.train,]
dat.test = dat[-i.train,]  # Remaining 50 samples for testing

# LDA (Linear Discriminant Analysis)
# Assumes equal covariance matrices across classes
lda.o = lda(Species~., data=dat.train)  # Fit LDA model using all features
lda.p = predict(lda.o, newdata=dat.test)  # Generate predictions on test set
names(lda.p)  # Available components in prediction object
(tb = table(lda.p$class, dat.test$Species))  # Confusion matrix
sum(diag(tb))/sum(tb)  # Classification accuracy

# QDA (Quadratic Discriminant Analysis) 
# More flexible than LDA - allows different covariance matrices per class
qda.o = qda(Species~., data=dat.train)
qda.p = predict(qda.o, newdata=dat.test)
(tb = table(qda.p$class, dat.test$Species))
sum(diag(tb))/sum(tb)

## (2) 3-class classification problem

dat = iris  # Original 3-class iris dataset
n = nrow(dat)
set.seed(4061)
dat = dat[sample(1:n),]  # Shuffle dataset

i.train = 1:100  # Same train-test split ratio
dat.train = dat[i.train,]
dat.test = dat[-i.train,]

# LDA for multi-class problem
lda.o = lda(Species~., data=dat.train)
lda.p = predict(lda.o, newdata=dat.test)
names(lda.p)
(tb = table(lda.p$class, dat.test$Species))
sum(diag(tb))/sum(tb)

# QDA for multi-class problem
qda.o = qda(Species~., data=dat.train)
qda.p = predict(qda.o, newdata=dat.test)
(tb = table(qda.p$class, dat.test$Species))
sum(diag(tb))/sum(tb)

# Model Comparison:
# 1. For the binary case, QDA often outperforms LDA because it can capture the 
#    non-linear boundary between virginica and the other species.
# 2. For the 3-class problem, QDA should theoretically perform better as it models
#    each class with its own covariance structure.
# 3. However, LDA might be more stable with limited data as it estimates fewer parameters.
# 4. The performance difference depends on how much the covariance matrices differ between classes.
# 5. QDA will tend to overfit with small training sets due to its higher complexity.
##############################################################################################################################
### Exercise 5: benchmarking
##############################################################################################################################
## (1) benchmarking on unscaled data

set.seed(4061)
n = nrow(Default)
dat = Default[sample(1:n, n, replace=FALSE), ]

# Creating train-validation split (70/30)
i.cv = sample(1:n, round(.7*n), replace=FALSE)
dat.cv = dat[i.cv,] # Cross-validation portion (70%)
dat.valid = dat[-i.cv,] # Hold-out validation set (30%)

# Setting hyperparameter for KNN
K.knn = 3 

# Set up k-fold cross-validation
K = 10  # Number of folds
N = length(i.cv)
folds = cut(1:N, K, labels=FALSE)  # Assign each observation to a fold

# Initialize vectors to store performance metrics
acc.knn = acc.glm = acc.lda = acc.qda = numeric(K)  # Accuracy
auc.knn = auc.glm = auc.lda = auc.qda = numeric(K)  # Area Under ROC Curve

# Cross-validation loop
for(k in 1:K){ 
	# Create train and test sets for this fold
	i.train	= which(folds!=k)
	dat.train = dat.cv[i.train, ]
	dat.test = dat.cv[-i.train, ]
	
	# Prepare data for KNN (requires special format)
	x.train = dat.train[,-1]  # All features (excluding response)
	y.train = dat.train[,1]   # Response variable
	x.test = dat.test[,-1]
	y.test = dat.test[,1]
	x.train[,1] = as.numeric(x.train[,1])  # Convert student status to numeric
	x.test[,1] = as.numeric(x.test[,1])
	
	# Train four different classification models
	knn.o = knn(x.train, x.test, y.train, K.knn)  # K-nearest neighbors 
	glm.o = glm(default~., data=dat.train, family=binomial(logit))  # Logistic regression
	lda.o = lda(default~., data=dat.train)  # Linear discriminant analysis
	qda.o = qda(default~., data=dat.train)  # Quadratic discriminant analysis
	
	# Generate class predictions
	knn.p = knn.o  # KNN already returns classifications
	glm.p = (predict(glm.o, newdata=dat.test, type="response") > 0.5)  # Convert probabilities to classes
	lda.p = predict(lda.o, newdata=dat.test)$class
	qda.p = predict(qda.o, newdata=dat.test)$class	
	
	# Create confusion matrices
	tb.knn = table(knn.p, y.test)
	tb.glm = table(glm.p, y.test)
	tb.lda = table(lda.p, y.test)
	tb.qda = table(qda.p, y.test)
	
	# Calculate and store accuracy for each model
	acc.knn[k] = sum(diag(tb.knn)) / sum(tb.knn)  # Proportion of correct classifications
	acc.glm[k] = sum(diag(tb.glm)) / sum(tb.glm)
	acc.lda[k] = sum(diag(tb.lda)) / sum(tb.lda)
	acc.qda[k] = sum(diag(tb.qda)) / sum(tb.qda)
	
	# ROC/AUC analysis requires probability outputs
	# WARNING: THIS IS NOT Pr(Y=1 | X), BUT Pr(Y = Y_hat | X):
	# KNN's "probability" is just the proportion of neighbors voting for the winning class,
	# NOT the probability of belonging to a specific class. This makes it unsuitable for ROC analysis!
	knn.p = attributes(knn(x.train, x.test, y.train, K.knn, prob=TRUE))$prob
	glm.p = predict(glm.o, newdata=dat.test, type="response")  # Probability of default
	lda.p = predict(lda.o, newdata=dat.test)$posterior[,2]  # Probability of being in second class
	qda.p = predict(qda.o, newdata=dat.test)$posterior[,2]
	
	# Calculate and store AUC (area under ROC curve)
	# auc.knn[k] = roc(y.test, knn.p)$auc  # Note: KNN prob isn't true probability, so this is commented out
	auc.glm[k] = roc(y.test, glm.p)$auc
	auc.lda[k] = roc(y.test, lda.p)$auc
	auc.qda[k] = roc(y.test, qda.p)$auc
}

# Visualize cross-validation results
boxplot(acc.knn, acc.glm, acc.lda, acc.qda,
	main="Overall CV prediction accuracy",
	names=c("kNN","GLM","LDA","QDA"))
	
boxplot(auc.glm, auc.lda, auc.qda,
	main="Overall CV AUC",
	names=c("GLM","LDA","QDA"))

# Additional analysis of final model performance 
plot(roc(y.test,knn.p))  # Plot ROC curve

# Get AUC value for logistic regression
roc(y.test, glm.p)$auc

# Detailed confusion matrix analysis with caret
library(caret)
(tb = table(y.test, glm.p>.5))
pred = as.factor(glm.p>.5)
pred = car::recode(pred, "FALSE='No'; TRUE='Yes'")  # Make factor levels match
caret::confusionMatrix(y.test, pred)  # Get detailed metrics
sum(diag(tb))/sum(tb)  # Calculate accuracy manually

# Model Comparison:
# 1. Logistic regression (GLM) typically performs well on this dataset as the 
#    default probability is mainly influenced by balance and income in a mostly linear way.
# 2. LDA makes stronger assumptions about normality but often performs similarly 
#    to logistic regression when those assumptions are reasonably met.
# 3. QDA can fit more complex decision boundaries but may overfit with limited data.
# 4. KNN is entirely non-parametric but sensitive to the choice of K and feature scaling.
# 5. The AUC metric better captures model performance for imbalanced datasets 
#    compared to accuracy, as it doesn't depend on a specific threshold.

##############################################################################################################################
### Exercise 6: benchmarking, again
############################################################################################################################## 

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





########################### My code for Exercise 6
# Get dataset ready
set.seed(4061)
n = nrow(Default)
dat = Default[sample(1:n, n, replace=FALSE), ]
i.cv = sample(1:n, round(.7*n), replace=FALSE)
x = dat.cv = dat[i.cv,]

# Fit model once
glmo = glm(default~., data=x, family=binomial)
predicted_probs = predict(glmo, type="response")
actual_class = x$default == "Yes"

# Use a wider range of thresholds including 0 and 1
thrs = c(0, seq(.01,.99,by=.01), 1)
sens = spec = numeric(length(thrs))

for(ithr in 1:length(thrs)) {
  thr = thrs[ithr]
  predicted_class = predicted_probs > thr
  
  # Calculate TP, FP, TN, FN
  TP = sum(predicted_class & actual_class)
  FP = sum(predicted_class & !actual_class)
  TN = sum(!predicted_class & !actual_class)
  FN = sum(!predicted_class & actual_class)
  
  # Calculate sensitivity and specificity
  sens[ithr] = TP/(TP + FN)
  spec[ithr] = TN/(TN + FP)
}

# Plot ROC curve
plot(1-spec, sens, type="l", 
     xlab="1-Specificity", ylab="Sensitivity", 
     main="ROC Curve", 
     xlim=c(0,1), ylim=c(0,1))
points(1-spec, sens, pch=20, cex=0.5)
abline(0, 1, lty=2)



###############################
# Evaluate a cross-validated ROC curve manually:

n = nrow(x)
K = 10
# Use a wider range of thresholds including 0 and 1
thrs = c(0, seq(.05,.95,by=.05), 1)
train.acc = test.acc = matrix(NA, nrow=K, ncol=length(thrs))
train.sens = train.spec = matrix(NA, nrow=K, ncol=length(thrs))
test.sens = test.spec = matrix(NA, nrow=K, ncol=length(thrs))
folds = cut(1:n, K, labels=FALSE)

for(ithr in 1:length(thrs)){
  thr = thrs[ithr]
  for(k in 1:K){
    itrain = which(folds!=k)
    glmo = glm(default~., data=x, 
               family=binomial,
               subset=itrain)
    
    # Training evaluations
    pred_train = glmo$fitted.values > thr
    actual_train = x$default[itrain] == "Yes"
    
    # Calculate confusion matrix components
    TP = sum(pred_train & actual_train)
    FN = sum(!pred_train & actual_train)
    TN = sum(!pred_train & !actual_train)
    FP = sum(pred_train & !actual_train)
    
    train.acc[k, ithr] = (TP + TN)/(TP + TN + FP + FN)
    train.sens[k, ithr] = TP/(TP + FN)
    train.spec[k, ithr] = TN/(TN + FP)
    
    # Test evaluations
    p.test = predict(glmo, x[-itrain,], type='response')
    pred_test = p.test > thr
    actual_test = x$default[-itrain] == "Yes"
    
    # Calculate test confusion matrix
    TP_test = sum(pred_test & actual_test)
    FN_test = sum(!pred_test & actual_test)
    TN_test = sum(!pred_test & !actual_test)
    FP_test = sum(pred_test & !actual_test)
    
    test.acc[k, ithr] = (TP_test + TN_test)/(TP_test + TN_test + FP_test + FN_test)
    test.sens[k, ithr] = TP_test/(TP_test + FN_test)
    test.spec[k, ithr] = TN_test/(TN_test + FP_test)
  }
}

# Calculate mean performance metrics
mean_test_sens = colMeans(test.sens)
mean_test_spec = colMeans(test.spec)

# Plot the cross-validated ROC curve
plot(1-mean_test_spec, mean_test_sens, type="l", 
     xlab="1-Specificity", ylab="Sensitivity", 
     main="Cross-validated ROC Curve",
     xlim=c(0,1), ylim=c(0,1))
points(1-mean_test_spec, mean_test_sens, pch=20)
abline(0, 1, lty=2)
