# --------------------------------------------------------
# ST4061 / ST6041 / ST6042
# 2024-25
# Eric Wolsztynski
# ...
# Exercises Section 2: Regularization
# --------------------------------------------------------

###############################################################
### Exercise 1: tuning LASSO
###############################################################

# Have a go at this exercise yourself...
# you can refer to ST4060 material:)

library(ISLR)
library(glmnet)
# ?glmnet

dat = na.omit(Hitters)
n = nrow(dat)
set.seed(4061)
dat = dat[sample(1:n, n, replace=FALSE),]
dat$Salary = log(dat$Salary)

x = model.matrix(Salary~., data=dat)[,-1]
y = dat$Salary

?cv.glmnet
?glmnet

############################################################### 
### Exercise 2: tuning LASSO + validation split
############################################################### 

# Have a go at this exercise yourself too...
# you can refer to ST4060 material:)

