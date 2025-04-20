 
## Question 1

#a)
#1 Young
#2 I, Y, I,I
#3 M, M, I,M
#4 M, I, M, I (Either mature or intermediate)
#5 Young

#We'll just assume #4 is mature

#b)
# True is Y,Y,I,M,Y
# I said  Y,I,M,M,Y
# So the misclassification rate here is 40% since I got 2 wrong


#c)
#i) 42
#ii) 29/42 = .69
#iii) 8/12 = .67 
#iv) 3/12 = .25



#Question 2

#a) Regression. The variable we are trying to predict is salary which is a continuous variable
#b) Elastic net model with half ridge, half lasso
#c) k fold cross validation. This is the definittion of how the function works. See ?cv.glmnet
#d) Crit 1 is the training set MSE of mod1 while Crit 2 is the training set MSE of mod2
#e) Crit 3 is the test set MSE of mod1 while Crit 4 is the test set MSE of mod2
#f) Train set median MSE median is lower than the test set median MSE as we expect. 
# the test set is testing the model on unseen points so the MSE will be higher whereas the train mse 
#is testing the model on the points used to train it so we expect a lower MSE
#g) The variance of crit 2 is about 60 times lower. Again as we mentioned before crit 2 is the model being
# tested against the points it was trained on the we get much better precision in our estimates which
# of course we dont see when testing on unseen data points

median(crit2)
median(crit4)
var(crit2)
var(crit4)



#### Question 3

require(tree)
require(randomForest)
require(caret)
require(e1071)

dat = read.csv("Question3_data.csv")
set.seed(4061)

newdata = dat[,-1]
corrplot::corrplot(cor(newdata))
#c1 and d3
#need to apply a correlation filter
corr_problems = findCorrelation(cor(newdata),names = T,verbose = F)
corr_problems

# Show excluded features
print("Features excluded by correlation filter:")
print(corr_problems)

#C Now perform univariate regression on this dataset
# So i want to exclude the features and then perform linear regressionon each predictor 
#on the dataset without the correlated features
# So remove D6 and C1 from dat
names(dat)
reg_dat = dat[,c(-12,-17)]
Y = reg_dat[,1]
p_value = list()
for(i in 2:ncol(reg_dat)){
  model = lm(Y ~ reg_dat[,i], data = reg_dat)
  p_value[i-1] = summary(model)$coefficients[2,4]
}

p_value
coef(lm(Y ~ reg_dat[,1], data = reg_dat))
options(scipen = 999)
p_adjusted = p.adjust(p_value, method = 'fdr')
which(p_adjusted <.05)

#So variables X1, X2, X3 are all good
features = names(reg_dat[,-1])
new_data_frame = data.frame(Features = features, P_Values = p_adjusted)
new_data_frame


#C) Fit a tree to the whole dataset
tree_model = tree(Y~.,dat)
summary(tree_model)
# Number of terminal nodes is 10
names(tree_model)
new_pred = predict(tree_model)
rmse_tree = sqrt(mean(new_pred- Y)^2)
rmse_tree
#rmse = sqrt(summary(tree_model)$dev)
#rmse

names(summary(tree_model))

#D) Prune the tree to 5 nodes
pruned_tree = prune.tree(tree_model,best = 5)
names(pruned_tree)
prune_pred = predict(pruned_tree)
rmse_prune = sqrt(mean(prune_pred- Y)^2)
rmse_prune


#E) Fit an SVM
SVM_model= svm(Y~.,data = dat)
SVM_model$kernel
names(SVM_model)

sqrt(mean(SVM_model$residuals)^2)
#i RMSE is 2.498449
#ii The kernel is a radial kernel

#f) Fit a random forest
rf_model = randomForest(Y~.,data =dat)
names(rf_model)
rf_model$mse #These are the oob MSE values
par(mfrow = c(1,1))
vi_rf = varImpPlot(rf_model)
importance(rf_model)

#RMSE
preds = predict(rf_model)
rmse_rf = sqrt(mean(preds - Y)^2)
rmse_rf


#g)
# c & d RMSE
rmse_tree
rmse_prune
# The pruned tree has a higher RMSE but we expect this as there is less overfitting. 
#It has less terminal nodes

# c & f RMSE
rmse_rf
rmse_tree
# Similar to above the random forest has less overfitting than the tree and thus a higher RMSE
# The random forest build many trees on bootstrapped subsets and thus showcases less RMSE






##############################################################
############# QUESTION 4
##############################################################
require(ISLR)
require(pROC)
Smarket
Smarket$Direction <- relevel(Smarket$Direction, ref = "Up") #Sets up as the positive class

x = Smarket[,-9]
y = Smarket$Direction

set.seed(4061)
train = sample(1:nrow(Smarket),1000)
test <- setdiff(1:nrow(Smarket), train)

x_train = x[train,]
y_train = y[train]
x_test = x[-train,]
y_test = y[-train]

rf_mod = randomForest(Direction~.,data = Smarket,subset = train)
names(rf_mod)
1 - sum(diag(rf_mod$confusion))/ sum(rf_mod$confusion)
#0.001022495

# (b) Generate a prediction of the 250 test observations form this random forest.
#Compute and plot the corresponding ROC. Quote the associated AUC.
df = data.frame(y_test,x_test)
rf_preds = predict(rf_mod, newdata = Smarket[test,], type = 'prob') #we want a prob for AUC and ROC
rf_preds
roc_rf = roc(y_test,rf_preds[,"Up"])
plot(roc_rf)
length(y_test)
length(rf_preds)



##c)

my_knn = knn(x_train, x_test, y_train, k =2)
knn.p = attributes(knn(x_train, x_test, y_train, 2, prob=TRUE))$prob
#knn.p[knn.p == .5] = 0
#knn.p
knn_roc = roc(y_test, knn.p )
plot(knn_roc, add= T, col= 'red', label = 'KNN ROC')
legend('topleft',legend = c("KNN ROC",'RF ROC'), lty = 1, col = c('red',"black"))
## an roc curve usualyy takes continuous values but here we only have discrete, even close
# to binary with just .5 and 1 values from the knn attributes

# attributes(knn(x.train, x.test, y.train, k, prob=TRUE))$prob
# does not return Pr(Y=1 | X), but Pr(Y = Y_hat | X)...
# Need to "hack" if we want Pr(Y=1 | X)...

knn_auc = roc(y_test,knn.p)$auc
knn_auc

#d)
set.seed(4061)
M = 1000
train = sample(1:nrow(Smarket), M)

#Compute test-set misclassification errors obtained from the kNN classifier for
#each value of k between 1 and 10. Plot this curve
mc = numeric(10)
for(k_tests in 1:10){
  knn_vals = knn(x_train, x_test, y_train, k =k_tests)
  tb = table(y_test, knn_vals)
  mc[k_tests] =  1- sum(diag(tb)) / sum(tb)
}

plot(1:10,mc)
