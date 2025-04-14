# --------------------------------------------------------
# ST4061 / ST6041 / ST6042
# 2024-25
# Eric Wolsztynski
# ...
# Exercises Section 1: Data pre-processing
# --------------------------------------------------------

library(glmnet)
library(survival)
library(ISLR)


###############################################################
### Example of scaling on PCA of time series 
###############################################################

plot(EuStockMarkets)
pca = prcomp(EuStockMarkets)$x
par(mfrow=c(4,1), mar=c(1,1,1,1))
for(i in 1:4){
	plot(pca[,i], lwd=4)
}

###############################################################
### Exercise 1: effect of scaling
###############################################################

dat = iris[1:100,]
dat$Species = droplevels(dat$Species)
x = dat[,1:4]
y = dat$Species
# we can also apply scaling to the x data directly:
dats = dat
dats[,1:4] = apply(dats[,1:4],2,scale)
xs = apply(x,2,scale)
pairs(x, col=c(1,4,2)[dats[,5]], pch=20, cex=2)
# (1)
pca.unscaled = prcomp(x) 
pca.scaled = prcomp(x,scale=TRUE)
pca.scaled.2 = prcomp(xs) # should be the same as pca.scaled

par(mfrow=c(1,3)) # scree plots
plot(pca.unscaled)
plot(pca.scaled)
plot(pca.scaled.2)

# plot the data on its first 2 dimensions in each space: 
par(mfrow=c(1,1))
plot(x[,1:2], cex=2, pch=20, col=y, main="Data in original space") 
biplot(pca.unscaled, main="Data in PCA space")
biplot(pca.scaled, main="Data in PCA space")
abline(h=0, col='orange')
abline(v=0, col='orange')
# see the binary separation in orange along PC1? 
# re-use this into biplot in original space:
pca.cols = c("blue","orange")[1+as.numeric(pca.unscaled$x[,1]>0)]
plot(x[,1:2], pch=20, col=pca.cols, cex=2,
	main="Data in original space\n colour-coded using PC1-split") 

par(mfrow=c(2,2), mar=c(3,3,3,3))
plot(pca.unscaled) # scree plot 
biplot(pca.unscaled) # biplot 
plot(pca.scaled) # scree plot 
biplot(pca.scaled) # biplot 
# now analyse this plot :)
par(mfrow=c(1,2))
biplot(pca.unscaled) # biplot 
biplot(pca.scaled) # biplot 

# (2)
logreg.unscaled = glm(Species~., data=dat) # make this work
logreg.unscaled = glm(Species~., data=dat, family='binomial')
logreg.scaled = glm(Species~., data=dats, family='binomial')
# discuss... 
# (... are the fits different?)
cbind(coef(logreg.unscaled), coef(logreg.scaled))

# (3)
x.m = model.matrix(Species~.+0, data=dat)
lasso.cv = cv.glmnet(x.m, y, family="binomial")
lasso.unscaled = glmnet(x.m, y, family="binomial", lambda=lasso.cv$lambda.min)
lasso.pred = predict(lasso.unscaled, newx=x.m, type="class")
#
xs.m = model.matrix(Species~.+0, data=dats)
lasso.cv = cv.glmnet(xs.m, y, family="binomial")
lasso.scaled = glmnet(xs.m, y, family="binomial", lambda=lasso.cv$lambda.min)
lasso.s.pred = predict(lasso.scaled, newx=xs.m, type="class")
#
cbind(coef(lasso.unscaled), coef(lasso.scaled))
table(lasso.pred, lasso.s.pred) # meh

###############################################################
### Exercise 2: data imputation
###############################################################

summary(lung)
boxplot(lung$meal.cal~lung$sex, col=c("cyan","pink"))
# can you think of other ways of analysing this?
wilcox.test(lung$meal.cal~lung$sex,)

# (1) lung cancer data: compare meal.cal values between male and female cohorts, 
# and discuss w.r.t. gender-specific data imputation 
# NB: "missing at random" vs "missing not at random"??

nas = is.na(lung$meal.cal) # track missing values
table(lung$sex)
table(nas, lung$sex)
imales = which(lung$sex==1)
m.all = mean(lung$meal.cal, na.rm=TRUE)
m.males = mean(lung$meal.cal[imales], na.rm=TRUE)
m.females = mean(lung$meal.cal[-imales], na.rm=TRUE)
t.test(lung$meal.cal[imales], lung$meal.cal[-imales])
# significant difference, hence must use different imputation
# values for each gender

# (2) Run Cox PHMs on original, overall imputed and gender-specific imputed
# datsets, using the cohort sample mean for data imputation. Compare and discuss 
# model fitting output.

dat1 = dat2 = lung
# impute overall mean in dat1:
inas = which(is.na(lung$meal.cal)) # track missing values
dat1$meal.cal[inas] = m.all 
# impute gender-specific mean in dat2:
dat2$meal.cal[which(is.na(lung$meal.cal) & (lung$sex==1))] = m.males
dat2$meal.cal[which(is.na(lung$meal.cal) & (lung$sex==2))] = m.females
cox0 = coxph(Surv(time,status)~.,data=lung) 
cox1 = coxph(Surv(time,status)~.,data=dat1) 
cox2 = coxph(Surv(time,status)~.,data=dat2)
summary(cox0)
summary(cox1)
summary(cox2)
# - dat1 and dat2 yield increased sample size (from 167 to 209, both imputed 
# datasets having 209 observations)
# - overall coefficient effects comparable between the 2 sets
# - marginal differences in covariate effect and significance between lung and {dat1;dat2}
# - no substantial difference between dat1 and dat2 outputs

###############################################################
### Exercise 3: data imputation
###############################################################

library(ISLR)
dat = Hitters

# (1) (Deletion)
sdat = na.omit(dat)
sx = model.matrix(Salary~.+0,data=sdat)
sy = sdat$Salary
cv.l = cv.glmnet(sx,sy)
slo = glmnet(sx,sy,lambda=cv.l$lambda.min)

# (2) Simple imputation (of Y) using overall mean
ina = which(is.na(dat$Salary))
dat$Salary[ina] = mean(dat$Salary[-ina])
x = model.matrix(Salary~.+0,data=dat)
y = dat$Salary
cv.l = cv.glmnet(x,y)
lo = glmnet(x,y,lambda=cv.l$lambda.min)

# (3)
slop = predict(slo,newx=sx)
lop = predict(lo,newx=x)
sqrt(mean((slop-sy)^2))
sqrt(mean((lop-y)^2))
plot(slop,lop[-ina])
abline(a=0,b=1,col=2,lwd=4)
abline(lm(lop[-ina]~slop), col='navy')

s1 = (slop-sy)
s2 = (lop-y)
par(mfrow=c(1,2))
plot(s1,ylim=range(s1)); abline(h=0)
plot(s2,ylim=range(s1)); abline(h=0)
sd(s1)
sd(s2)

# What could we do instead of imputing the Y?

###############################################################
### Exercise 4: resampling
###############################################################
########################
###### Exercise 4 ######
########################
# Load dataset and inspect
data(trees)
?trees

# Set seed for cross-validation procedure
set.seed(4060)
k = 10
n = nrow(trees)

# Create folds for disjoint splits
folds = sample(rep(1:k, length.out = n)) #each data point is assigned a fold

# Vectors to store slope estimates (beta coefficient)
betas = numeric(k)

# 1. 10-fold CV on the original dataset
for(i in 1:k){
  test_ind = which(folds == i)
  train_data = trees[-test_ind, ]
  # Fit linear model on the training set
  my_model = lm(Height ~ Girth, data = train_data)
  # Store the slope (coefficient for Girth)
  betas[i] = my_model$coefficients[2]
}

cv_slope_estimate = mean(betas)
cat("Cross-validated slope estimate (original):", cv_slope_estimate, "\n")

# 2. 10-fold CV on the randomised dataset
# First, shuffle the dataset as instructed
set.seed(1)
data_new = trees[sample(1:n), ]

# Create disjoint folds for the shuffled dataset as well (we can re-use the same folds or generate new ones)
# Here we generate new folds:
folds_shuffle = sample(rep(1:k, length.out = n))

betas_shuffle = numeric(k)
for(i in 1:k){
  test_ind = which(folds_shuffle == i)
  train_data = data_new[-test_ind, ]
  shuffle_model = lm(Height ~ Girth, data = train_data)
  betas_shuffle[i] = shuffle_model$coefficients[2]
}

cv_slope_estimate_shuffle = mean(betas_shuffle)
cat("Cross-validated slope estimate (shuffled):", cv_slope_estimate_shuffle, "\n")

# 3. Compare sampling distributions using boxplots
par(mfrow = c(1,2), mar = c(3,3,3,2))
boxplot(betas, betas_shuffle, 
        names = c("Beta (Original)", "Beta (Shuffled)"),
        main = "Distribution of Slope Estimates")

par(mfrow = c(1,1))


## Perform two-sided t-test
t_test = t.test(betas, betas_shuffle, alternative = "two.sided")
F_test = var.test(betas, betas_shuffle,alternative =  "two.sided")

t_test
F_test

#p-value very high --> no statistical significance

#F-test
#The point estimate of the variance ratio is 2.2154;
#however, the p-value of 0.2517 indicates that this difference is not 
#statistically significant at conventional levels. 
#The wide confidence interval (ranging from about 0.55 to 8.92) tells us 
#that there is substantial uncertainty about the true ratio of variances. 
#In other words, even though the point estimate is greater than 1, 
#the data do not provide strong evidence that the variances of the slope estimates are different


#t-test
#The extremely high p-value (0.9972) indicates that there is virtually no difference in the means of 
#the two distributions. 
#The confidence interval for the difference in means includes 0 and is very tight, suggesting that 
#the average slope estimates are essentially identical whether you 
#use the original or the shuffled dataset for cross-validation.


###############################################################
### Exercise 5: resampling (CV vs bootstrapping)
###############################################################

# Implement this simple analysis and discuss - think about 
# (sub)sample sizes!

x = trees$Girth   # sorted in increasing order...
y = trees$Height
plot(x, y, pch=20)
summary(lm(y~x))
N = nrow(trees)

# (1) 10-fold CV on original dataset
set.seed(4060)
K = 10
cc = numeric(K)
folds = cut(1:N, K, labels=FALSE)
for(k in 1:K){
	i = which(folds==k)
	# train:
	lmo = lm(y[-i]~x[-i])
	cc[k] = summary(lmo)$coef[2,2]
	# (NB: no testing here, so not the conventional use of CV)
}
mean(cc)

# (2) 10-fold CV on randomized dataset
set.seed(1)
mix = sample(1:nrow(trees), replace=FALSE)
xr = trees$Girth[mix]
yr = trees$Height[mix]
set.seed(4060)
K = 10
ccr = numeric(K)
folds = cut(1:N, K, labels=FALSE)
for(k in 1:K){
	i = which(folds==k)
	lmo = lm(yr[-i]~xr[-i])
	ccr[k] = summary(lmo)$coef[2,2]
}
mean(ccr)

sd(ccr)
sd(cc)
boxplot(cc,ccr)
t.test(cc,ccr)
var.test(cc,ccr)

# (3) Bootstrapping (additional note)
set.seed(4060)
K = 100
cb = numeric(K)
for(i in 1:K){
	# bootstrapping
	ib = sample(1:N,N,replace=TRUE)
	lmb = lm(yr[ib]~xr[ib])
	cb[i] = summary(lmb)$coef[2,2]
}
mean(cb)

dev.new()
par(font=2, font.axis=2, font.lab=2)
boxplot(cbind(cc,ccr,cb), names=c("CV","CVr","Bootstrap"))
abline(h=1.0544)
t.test(cc,cb)
# Explain why these are different?

round(cc, 3)

# ------------------------------------
set.seed(1)
N = 1000
x = runif(N, 2, 20)
y = 2 + 5*x + 5*rnorm(N)
plot(x,y,pch=20)
o = lm(y~x)

# Repeated (33x) 3-fold CV 
set.seed(4061)
R = 33
K = 3
cvr = numeric(K*R)
for(r in 1:R){
	ir = sample(1:N,replace=FALSE)
	xr = x[ir]
	yr = y[ir]
	folds = cut(1:N, K, labels=FALSE)
	for(k in 1:K){
		i = which(folds==k)
		df = data.frame(x=xr[-i],y=yr[-i]) # train data
		dft = data.frame(x=xr[i],y=yr[i]) #Â test data
		lmo = lm(y~.,data=df)
		yp = predict(lmo,newdata=dft)
		rmse = (sqrt(mean((yp-yr[i])^2)))
		cvr[(k+(r-1)*K)] = rmse
	}
}

# 100 x Boot
set.seed(4061)
R = 100
bsr = numeric(R)
for(r in 1:R){
	i = sample(1:N,replace=TRUE)
	xr = x[i]
	yr = y[i]
	df = data.frame(x=xr[i],y=yr[i]) # train data
	dft = data.frame(x=xr[-i],y=yr[-i]) #Â test data (OOB)
	lmo = lm(y~.,data=df)
	yp = predict(lmo,newdata=dft)
	rmse = (sqrt(mean((yp-yr[-i])^2)))
	bsr[r] = rmse
}

boxplot(cvr,bsr,names=c())

