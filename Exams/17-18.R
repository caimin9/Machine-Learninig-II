#2017-18 Paper
#Question1 
M =100
set.seed(4061)
dat = iris[sample(1:nrow(iris)),]
dat[,1:4] = apply(dat[,1:4],2,scale)
itrain = sample(1:nrow(iris), M)

#Grow a classification tree
library(tree)
my_tree = tree(Species~.,data = dat, subset = itrain, split = 'gini')
my_tree
summary(my_tree)$size #number of terminal nodes

y_test = dat[-itrain,'Species'] 
y_train = dat[itrain, 'Species']
y_fitted = predict(my_tree, newdata = dat[itrain,], type ='class')
tb = table(y_fitted, y_train)

1 - sum(diag(tb))/sum(tb)

#useful variables are petal width, petal length, sepal.length


#b Petal width and petal length

#c
# 2×2 panel of boxplots for the four iris predictors
par(mfrow = c(2, 2),        # 2 rows, 2 columns
    mar   = c(5, 4, 2, 1))  # margins: bottom, left, top, right

# 1. Sepal Length
boxplot(Sepal.Length ~ Species, data = iris,
        main = "Sepal Length",
        xlab = "Species",
        ylab = "Length (cm)")

# 2. Sepal Width
boxplot(Sepal.Width ~ Species, data = iris,
        main = "Sepal Width",
        xlab = "Species",
        ylab = "Width (cm)")

# 3. Petal Length
boxplot(Petal.Length ~ Species, data = iris,
        main = "Petal Length",
        xlab = "Species",
        ylab = "Length (cm)")

# 4. Petal Width
boxplot(Petal.Width ~ Species, data = iris,
        main = "Petal Width",
        xlab = "Species",
        ylab = "Width (cm)")

# Reset graphics parameters if needed:
# par(mfrow = c(1,1))


#d Based on the plots comment on answers form b
# the disparity per class in petal width and petal length is much greater
# in comparison to speal width and sepal length thuis making it easier for the 
# tree to establish a separation in classes


#e)
preds = predict(my_tree, newdata = dat[-itrain,], type = 'class')
tb2 = table(preds,y_test)
1 - sum(diag(tb2))/sum(tb2)


#f) CV prune 
pruned_tree = cv.tree(my_tree,FUN = prune.tree)
opt_size = pruned_tree$size[which.min(pruned_tree$dev)]
#optimal size is 5


#g) Train a random forest
library(randomForest)
rf = randomForest(Species~.,data = dat, subset = itrain)
#oob error estimate rate is 7%

#h)

rf_preds =  predict(rf , newdata = dat[-itrain,])
tb3 = table(rf_preds,y_test)
1 - (sum(diag(tb3))/sum(tb3))
#pred error rate is 66%
#i
#Can't generate ROC curve for this tree since we have a 3 class problem


############################################################
###### Question 2
############################################################

dat = model.matrix(Apps~.,College)[,-1]
dat = apply(dat,2,scale)
set.seed(4061)
itrain = sample(1:nrow(dat),500)
x_train = dat[itrain,]
x_test = dat[-itrain,]

#a) regression or classification
#This is a regression probelm
y =College$Apps
y_train = y[itrain]
y_test = y[-itrain]

#b
library(glmnet)
# Simple fit with fixed lambda
model <- glmnet( x_train ,y_train , alpha = 1)
# Cross - validation for lambda selection
cv_model <- cv.glmnet(x_train, y_train , alpha = 1)
best_lambda <- cv_model$lambda.min
model <- glmnet( x_train , y_train , alpha = 1 , lambda = best_lambda )

ridge_coef = predict(model, type = 'coefficients', s = best_lambda)[1:18,]


#c 
set.seed(4061)
#i 
preds = predict(model, newx = x_test)

rmse = sqrt(mean(preds - y_test)^2)

#ii
cor(preds,y_test)

#iii
plot(y_test,preds)

#d)
library(randomForest)
rf.mod5 = randomForest(x_train,y_train,mtry = 5)
rf.mod15 = randomForest(x_train,y_train ,mtry = 15)

summary(rf.mod5)

# Training errors for rf.mod5 and rf.mod15
rf5_train_preds = predict(rf.mod5, x_train)
rf15_train_preds = predict(rf.mod15, x_train)
rf5_train_rmse = sqrt(mean((rf5_train_preds - y_train)^2))
rf15_train_rmse = sqrt(mean((rf15_train_preds - y_train)^2))
print(c(rf5_train_rmse, rf15_train_rmse))


#e
# Test predictions with rf.mod15
rf15_test_preds = predict(rf.mod15, x_test)
rf15_test_rmse = sqrt(mean((rf15_test_preds - y_test)^2))
print(rf15_test_rmse)
# Compare with LASSO RMSE to determine which is better
print(paste("LASSO RMSE:", rmse))
print(paste("RF15 RMSE:", rf15_test_rmse))

importance(rf.mod5,scale = T)
varImp(rf.mod15,scale = F)
# I think based on these results the relationship might be more linear 
# between college APPS and the other variables in here



#Question 3
# Load libraries
library(ISLR)

library(randomForest)
library(class)
library(pROC)

# Setup data
x = Smarket[,-9]
y = Smarket$Direction
set.seed(4061)
train = sample(1:nrow(Smarket),1000)

# a) Fit random forest and get training error
rf_model = randomForest(x[train,], y[train], ntree=500)
train_pred = predict(rf_model)
train_error = mean(train_pred != y[train])
print(paste("Random Forest training error:", train_error))

# b) Test predictions and ROC curve
rf_probs = predict(rf_model, newdata=x[-train,], type="prob")[,"Up"]
rf_roc = roc(y[-train], rf_probs)
rf_auc = auc(rf_roc)
plot(rf_roc, main="ROC Curves", col="blue")
print(paste("Random Forest AUC:", rf_auc))

# c) kNN classifier with k=2
# Need to standardize predictors for kNN
x_scaled = scale(x)
knn_pred = knn(train=x_scaled[train,], test=x_scaled[-train,], 
               cl=y[train], k=2, prob=TRUE)
# Extract probability attributes
knn_probs = attributes(knn_pred)$prob
# If "Down" is the positive class in the result, invert probabilities
if(knn_pred[1] == "Down") {
  knn_probs = 1 - knn_probs
}
knn_roc = roc(y[-train], knn_probs)
knn_auc = auc(knn_roc)
plot(knn_roc, add=TRUE, col="red")
legend("bottomright", legend=c("Random Forest", "kNN (k=2)"), 
       col=c("blue", "red"), lwd=2)
print(paste("kNN (k=2) AUC:", knn_auc))

# d) kNN error rates for different k values
set.seed(4061)
M = 1000
train = sample(1:nrow(Smarket), M)

error_rates = numeric(10)
for(k in 1:10) {
  knn_pred = knn(train=x_scaled[train,], test=x_scaled[-train,], 
                 cl=y[train], k=k)
  error_rates[k] = mean(knn_pred != y[-train])
}

plot(1:10, error_rates, type="b", xlab="k", ylab="Test Error Rate",
     main="kNN Test Error Rates by k Value", pch=19)
