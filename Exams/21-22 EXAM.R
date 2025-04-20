#2021-2022 Exam Paper

#Question 1
#i) 
#A tree

#ii) 
#A classification problem since the Y variable is categorical data and not numerical

#iii) 
# obs1: Young
# obs2: Intermediate 
# obs3: Mature
# obs4: Mature
# obs5: Young

#iv
# 2 is wrong, 3 is wrong, 
# so misclassification rate is 2/5 = 40%

#b)
# this is bagging (bootstrapping and aggregating)

#c)
# i think this is a random forest

#d)
# random forest from part c would decorrelate our trees by only choosing a subset of m = 2 predictors at
# each split. This would prevent overfitting



#Question 2
dat.nona = na.omit(airquality)
dat = airquality
dat.nona$Month = as.factor(dat.nona$Month)
dat.nona$Day = as.factor(dat.nona$Day)
dat$Month = as.factor(dat$Month)
dat$Day = as.factor(dat$Day)

set.seed(4060)
par(mfrow = c(1,1))
cor_matrix = cor(na.omit(airquality))
corrplot::corrplot(cor_matrix)
pairs(airquality)
#So temp and ozone and wind and ozone look highly correlated
qqplot(scale(dat$Wind), scale(dat$Ozone))
dat
library(DataExplorer)
plot_bar(dat)


#b)
# We need to impute data for both Solar and Ozone
# Ozone is correlated with temp so that should be considered
# Solar has not strong correlations with any variables so maybe consider using
# the median of solar for imputation

#c)
indices = which(is.na(dat$Solar.R)) 
solar_med = median(dat.nona$Solar.R)

dat[indices,'Solar.R'] = solar_med

glm_fit = glm(Wind~., data = dat)
sum(residuals(glm_fit)^2)
#614.4613

#d)
quad_model = lm(Ozone~Temp + I(Temp)^2,data = dat.nona)
index = which(is.na(dat$Ozone))
predictions = predict(quad_model, newdata = dat[index,])
dat$Ozone[index] = predictions

#new glm_fit
glm_final = glm(Wind~., data = dat)
RSS_glm_final = sum((glm_final$residuals)^2)
RSS_glm_final


#e)
library(tree)
my_tree = tree(Wind~.,data = dat)
plot(my_tree)
text(my_tree)
summary(my_tree)$used
sum(residuals(my_tree)^2)

#i Variables used were Ozone, Day, Month, Temp
#ii Plot done above
#iii 549.7572 

#f)
pruned_tree= prune.tree(my_tree, best =10)
pruned_tree
summary(pruned_tree)$used
sum(residuals(pruned_tree)^2)

#i Used Ozone, Day, Month, Temp
#ii 664.5364

#g
# The residuals increasing in the pruned tree makes sense since it prevents 
# overfitting




# Question 3
library(ISLR)
library(gbm)
library(randomForest)


x.train = Khan$xtrain
x.test = Khan$xtest
y.train = as.factor(Khan$ytrain)
y.test = as.factor(Khan$ytest)
set.seed(4061)

t1 = table(y.train)
t2 = table(y.test)
my_table = rbind(TrainSet = t1,TestSet = t2)
my_table
sum(my_table[2,1]/sum(my_table[1,1]))
sum(my_table[2,2]/sum(my_table[1,2]))
sum(my_table[2,3]/sum(my_table[1,3]))
sum(my_table[2,4]/sum(my_table[1,4]))
sum(my_table[1,])/sum(my_table)
#As you can see the ratios of the splits of each class differ. Half of class 3
# lies in the test set but only a quarter of class 4 lies in the test set.
# We've done a 76/24 split on our train/test sets so roughly 25% of each class
# Should lie in the test set. this is true for class 2 and 4 but not 1 and 3

#b) This is a classification problem

#c) Quote the corresponding confusion matrix 
my_rf= randomForest(x = x.train, y = y.train)
my_rf$confusion

#d Generate predictions for the test data from the rf model fit
my_preds = predict(my_rf, newdata = x.test)
confusion_matrix = table(y.test,my_preds)
confusion_matrix
sum(diag(confusion_matrix))/ sum(confusion_matrix)
# It's 95%

#e) 
ind = which(importance(my_rf) >.4)
importance(my_rf)[ind,]
#187, 246, 545, 1003, 1389, 1954, 2050
varImpPlot(my_rf)


#f)
# the variable importance tells us the reduction in the models accuracy when
# The info from that variable is stripped from the model, or for the gini coefficient

# MeanDecreaseAccuracy
# How much the model's out-of-bag (OOB) accuracy drops if you randomly permute ("break") that
# variable.
# A large decrease ??? the variable was carrying critical information for correct c
# lassification.

# MeanDecrease Gini
# Sum of the reduction in Gini impurity (for classification) across all splits where that 
# variable is used.
# Higher ??? splitting on that variable yields purer child nodes more often, 
# so it helps distinguish classes

#g)
my_gbm = gbm(y.train~., data = data.frame(x.train))
#predictions = predict(my_gbm, n.trees = my_gbm$n.trees)
gbm_preds = predict(my_gbm,newdata = data.frame(x.test), n.trees = my_gbm$n.trees)
gp = apply(gbm_preds,1,which.max)
tb1 = table(gp, y.test)
sum(diag(tb1))/ sum(tb1)
#95% accuracy



########################################################################
############ Question 4
########################################################################

X = read.csv(file="Q4_X.csv", header=TRUE)
Y = read.csv(file="Q4_Y.csv", header=FALSE, stringsAsFactors=TRUE)[,1]
X.valid = read.csv(file="Q4_Xvalid.csv", header=TRUE)
Y.valid = read.csv(file="Q4_Yvalid.csv", header=FALSE,
                   stringsAsFactors=TRUE)[,1]




