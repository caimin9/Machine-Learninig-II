# 2019-20 PAPER

#Question 1(a)  
#i.) The formula describes an additive model of weak learners multiplied by weights, with an explicit shrinkage factor. 
#In practice, this is implemented by Friedman's gradient-boosting machine: 
#at each step, a weak learner T(X) is fitted to the negative gradient and then added into the ensemble.  

#ii. The parameter lambda is the shrinkage (or learning-rate) parameter.
#It scales down the contribution of each newly fitted weak learner-controlling the step-size in the
#functional gradient-descent-which helps to regularise the ensemble and prevent over-fitting.  

#Question 1(b)  
#Figure1 shows the printed output from fitting a neural network using the neuralnet package in R. From that output:  
#- Model name: a feedforward neural network (multilayer perceptron)  
#- Number of predictors used: ten (variables X1 through X10)  
#- Number of hidden layers: one  
#- Size of the hidden layer: two neurons  

#Question 1(c)  
#Figure2 displays two bar-plots of mean decrease in Gini index for the same set of predictors, under two different ensemble methods:  
#- Ensemble method 1 (left plot): most likely a random forest, which uses a random subset of predictors at each split  
#- Ensemble method 2 (right plot): most likely bagging, which uses all predictors at every split  

#Both plots rank variables X9, X10, ., X5 by their importance. 
#The bagged trees show larger absolute importance values-because every split considers all predictors, 
#so the strongest predictors are used more often and drive down Gini more. The random forest's feature-subsampling reduces correlation 
#among trees and slightly lowers each variable's average importance; it can also change the relative ranking of predictors when some are 
#highly correlated.


#QUESTION 2
# Load required packages
library(MASS)      # for Boston dataset
library(glmnet)    # for ridge regression

# 2(a) Number of predictors P (excluding the response medv)
dat <- Boston
P <- ncol(dat) - 1
print(P)  # Expect 13

# 2(b) Proportion of distinct observations in each bootstrap sample
set.seed(4061)
B      <- 100
n      <- nrow(dat)
prop_u <- numeric(B)

for (b in seq_len(B)) {
  idx       <- sample(seq_len(n), size = n, replace = TRUE)
  prop_u[b] <- length(unique(idx)) / n
}

# Estimated average proportion of unique points
mean(prop_u)

# Plot: histogram of unique-point proportions
hist(prop_u,
     breaks = 20,
     main   = "Proportion of Unique Obs. per Bootstrap Sample",
     xlab   = "Proportion Unique",
     ylab   = "Frequency")

# 2(c) Bootstrap OLS + OOB RMSE
set.seed(4061)
rmse_ols <- numeric(B)

for (b in seq_len(B)) {
  idx_b   <- sample(seq_len(n), size = n, replace = TRUE)
  oob_idx <- setdiff(seq_len(n), unique(idx_b))
  
  train   <- dat[idx_b, ]
  test    <- dat[oob_idx, ]
  
  fit_ols <- lm(medv ~ ., data = train)
  preds   <- predict(fit_ols, newdata = test)
  
  rmse_ols[b] <- sqrt(mean((test$medv - preds)^2))
}

# (i) Final RMSE estimate
rmse_ols_mean <- mean(rmse_ols)
print(rmse_ols_mean)

# (ii) Standard error of the RMSE estimate
rmse_ols_se   <- sd(rmse_ols) / sqrt(B)
print(rmse_ols_se)

# 2(d) Bootstrap ridge (?? = 0.5) + OOB RMSE
set.seed(4061)
rmse_ridge <- numeric(B)

for (b in seq_len(B)) {
  idx_b   <- sample(seq_len(n), size = n, replace = TRUE)
  oob_idx <- setdiff(seq_len(n), unique(idx_b))
  
  train   <- dat[idx_b, ]
  test    <- dat[oob_idx, ]
  
  # Prepare design matrices
  x_train <- model.matrix(medv ~ ., data = train)[, -1]
  y_train <- train$medv
  x_test  <- model.matrix(medv ~ ., data = test)[, -1]
  
  # Fit ridge with fixed lambda = 0.5
  fit_rid <- glmnet(x_train, y_train,
                    alpha     = 0,
                    lambda    = 0.5,
                    standardize = TRUE)
  
  preds_r <- predict(fit_rid, newx = x_test, s = 0.5)
  
  rmse_ridge[b] <- sqrt(mean((test$medv - preds_r)^2))
}

# (i) Final ridge RMSE estimate
rmse_ridge_mean <- mean(rmse_ridge)
print(rmse_ridge_mean)

# (ii) Boxplot comparing OLS vs Ridge RMSE distributions
boxplot(rmse_ols, rmse_ridge,
        names = c("OLS", "Ridge (??=0.5)"),
        main  = "OOB RMSE: OLS vs Ridge",
        ylab  = "RMSE")



#############################################################################
# Question 3: Classification tasks in R (no caret)
#############################################################################
# Load required packages
library(randomForest)  # for random forest & bagging
library(glmnet)        # for lasso
library(pROC)          # for ROC analysis

# Load data
x         <- read.csv("Q3_x.csv")
y         <- read.csv("Q3_y.csv")[,1]
x.valid   <- read.csv("Q3_x_valid.csv")
y.valid   <- read.csv("Q3_y_valid.csv")[,1]

# Ensure response is a factor for classification
y   <- factor(y)
y.valid <- factor(y.valid)

# Part (a): Random Forest
set.seed(4061)
rf.fit <- randomForest(x = x, y = y)

# (i) Number of variables tried at each split
print(rf.fit$mtry)

# (ii) Variable importance plot
varImpPlot(rf.fit)

# If needed, top 5 variables by MeanDecreaseGini:
imp.rf <- importance(rf.fit, type = 2)
head(sort(imp.rf[,1], decreasing = TRUE), 5)

# Part (b): Bagging (mtry = all predictors)
p <- ncol(x)
set.seed(4061)
bag.fit <- randomForest(x = x, y = y, mtry = p)

# (i) mtry used at each split
print(bag.fit$mtry)

# (ii) Variable importance plot for bagging
varImpPlot(bag.fit)

# Top 5 for bagging
imp.bag <- importance(bag.fit, type = 2)
head(sort(imp.bag[,1], decreasing = TRUE), 5)

# (iii) Compare magnitude of top importance
top.rf  <- head(sort(imp.rf[,1],  decreasing = TRUE), 1)
top.bag <- head(sort(imp.bag[,1], decreasing = TRUE), 1)
print(top.rf)
print(top.bag)

# Part (c): LASSO model
set.seed(4061)
# Prepare design matrices
X <- model.matrix(~ ., data = x)[, -1]
Y <- y

# cv.glmnet for binomial lasso
cv.lasso <- cv.glmnet(X, Y, family = "binomial", alpha = 1)

# (i) Optimal lambda
lambda.opt <- cv.lasso$lambda.min
print(lambda.opt)

# (ii) Coefficients at lambda.opt
coef.lasso <- coef(cv.lasso, s = "lambda.min")
print(coef.lasso)

# (iii) Top 5 absolute coefficients (excluding intercept)
coefs     <- as.matrix(coef.lasso)[-1, , drop = FALSE]
top5lasso <- sort(abs(coefs), decreasing = TRUE)[1:5]
print(top5lasso)

# Part (d): Backward stepwise logistic regression
# Fit full model
full.glm <- glm(y ~ ., data = data.frame(x, y = y), family = binomial)

# Backward selection based on AIC
step.glm <- step(full.glm, direction = "backward", trace = FALSE)

# Final model summary
summary(step.glm)

# Part (e): Simple correlation analysis for selected variables
vars      <- c("DiscMM", "PctDiscMM", "PriceDiff", "SalePriceMM")
cor.mat   <- cor(x[, vars])
print(cor.mat)

# Part (f): Assess importance of PriceDiff across models
# We already have its importance in rf and bag, and its coefficient in lasso & step.glm
# (User to interpret these values manually)

# Part (g): Validation predictions & ROC

# (i) Confusion matrix for Random Forest on validation set
rf.pred.prob <- predict(rf.fit, newdata = x.valid, type = "prob")[,2]
rf.pred      <- factor(ifelse(rf.pred.prob > 0.5, levels(y)[2], levels(y)[1]),
                       levels = levels(y))
table("RF Confusion" = rf.pred, "Truth" = y.valid)

# (ii) Confusion matrix for LASSO on validation set
X.valid      <- model.matrix(~ ., data = x.valid)[, -1]
lasso.pred.prob <- predict(cv.lasso, newx = X.valid, s = "lambda.min", type = "response")
lasso.pred      <- factor(ifelse(lasso.pred.prob > 0.5, levels(y)[2], levels(y)[1]),
                          levels = levels(y))
table("LASSO Confusion" = lasso.pred, "Truth" = y.valid)

# (iii) ROC curves in same plot
roc.rf    <- roc(response = y.valid, predictor = rf.pred.prob)
roc.lasso <- roc(response = y.valid, predictor = as.numeric(lasso.pred.prob))
plot(roc.rf, col = "black", main = "ROC Curves: RF (black) vs LASSO (red)")
lines(roc.lasso, col = "red")
legend("bottomright",
       legend = c(sprintf("RF AUC = %.3f", auc(roc.rf)),
                  sprintf("LASSO AUC = %.3f", auc(roc.lasso))),
       col = c("black", "red"), lwd = 2)

# (iv) AUC values
print(auc(roc.rf))
print(auc(roc.lasso))

# (v) Limitation: 
# This code does not implement any form of cross-validation or repeated splitting
# for validation. A more reliable approach would be to use k-fold CV or repeated
# train/validation splits to stabilise performance estimates.

