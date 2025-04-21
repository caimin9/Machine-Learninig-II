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



####################### My code #########################
library(ISLR)
library(glmnet)
ISLR::Hitters
dat = na.omit(Hitters)
lam_vals = 10^seq(10,-2,length =100)
n = nrow(dat)
set.seed(4061)
dat = dat[sample(1:n, n, replace=FALSE),]
dat$Salary = log(dat$Salary)

x = model.matrix(Salary~.,data = dat)
y = hits$Salary

lasso = glmnet(x,y, alpha = 1, lambda = lam_vals)
ridge = glmnet(x,y,alpha= 0, lambda = lam_vals)


small_lambda = lam_vals[100]
ridge_coef = predict(ridge, type = 'coefficients', s = small_lambda)[1:20,]
ridge_coef
lasso_coef = predict(lasso, type = 'coefficients', s = small_lambda)[1:20,]
lasso_coef


comparison = data.frame(
  Variable = names(ridge_coef),
  Ridge = as.numeric(ridge_coef),
  LASSO = as.numeric(lasso_coef)
)
print(comparison)


library(ggplot2)

# Calculate absolute difference
comparison$Diff = abs(comparison$Ridge - comparison$LASSO)

# Filter out the intercept for plotting
comparison_noint = subset(comparison, Variable != "(Intercept)")

# Sort by difference
comparison_noint = comparison_noint[order(-comparison_noint$Diff), ]

# Plot top 15 most different coefficients
ggplot(head(comparison_noint, 15), aes(x = reorder(Variable, Diff))) +
  geom_col(aes(y = Ridge), fill = "steelblue", alpha = 0.7) +
  geom_col(aes(y = LASSO), fill = "tomato", alpha = 0.5) +
  coord_flip() +
  labs(title = "Top 15 Variables: Ridge vs LASSO Coefficients",
       y = "Coefficient Value", x = "Variable") +
  theme_minimal()




library(tidyr)

# Reshape data for grouped bar plot
comparison_long = comparison_noint |>
  head(15) |>
  pivot_longer(cols = c("Ridge", "LASSO"), names_to = "Model", values_to = "Coefficient")

ggplot(comparison_long, aes(x = reorder(Variable, abs(Coefficient)), y = Coefficient, fill = Model)) +
  geom_col(position = "dodge") +
  coord_flip() +
  labs(title = "Top 15 Variables: Ridge vs LASSO Coefficients",
       y = "Coefficient Value", x = "Variable") +
  theme_minimal()



library(ggplot2)

ggplot(head(comparison_noint, 15), aes(x = Ridge, xend = LASSO, y = reorder(Variable, Diff))) +
  geom_segment(aes(yend = Variable), colour = "grey60") +
  geom_point(aes(colour = "Ridge"), size = 3) +
  geom_point(aes(x = LASSO, colour = "LASSO"), size = 3) +
  labs(title = "Ridge vs LASSO Coefficients (Shrinkage Comparison)",
       x = "Coefficient Value", y = "Variable") +
  scale_colour_manual(values = c("Ridge" = "steelblue", "LASSO" = "tomato")) +
  theme_minimal()



############################################################### 
### Exercise 2: tuning LASSO + validation split
############################################################### 

# Have a go at this exercise yourself too...
# you can refer to ST4060 material:)

### my code


library(ISLR)
library(glmnet)
library(dplyr)

# Clean and prep data
Hitters = na.omit(Hitters)
set.seed(123)  # for reproducibility

# Train/test split: 70/30
n = nrow(Hitters)
train_idx = sample(1:n, size = 0.7 * n)
train = Hitters[train_idx, ]
test = Hitters[-train_idx, ]

# Create model matrices (glmnet needs numeric matrix)
X_train = model.matrix(Salary ~ ., train)[, -1]
y_train = train$Salary

X_test = model.matrix(Salary ~ ., test)[, -1]
y_test = test$Salary


# Lambda grid
lambda_grid = 10^seq(10, -2, length = 100)

# ---- Ridge regression ----
cv_ridge = cv.glmnet(X_train, y_train, alpha = 0, lambda = lambda_grid)
best_lambda_ridge = cv_ridge$lambda.min
ridge_model = glmnet(X_train, y_train, alpha = 0, lambda = best_lambda_ridge)
ridge_pred = predict(ridge_model, s = best_lambda_ridge, newx = X_test)

# ---- LASSO regression ----
cv_lasso = cv.glmnet(X_train, y_train, alpha = 1, lambda = lambda_grid)
best_lambda_lasso = cv_lasso$lambda.min
lasso_model = glmnet(X_train, y_train, alpha = 1, lambda = best_lambda_lasso)
lasso_pred = predict(lasso_model, s = best_lambda_lasso, newx = X_test)

# ---- Ordinary Least Squares ----
ols_model = lm(Salary ~ ., data = train)
ols_pred = predict(ols_model, newdata = test)


# Mean squared error function
mse = function(actual, pred) mean((actual - pred)^2)

# Compute test MSEs
ridge_mse = mse(y_test, ridge_pred)
lasso_mse = mse(y_test, lasso_pred)
ols_mse   = mse(y_test, ols_pred)

# Combine into table
results = data.frame(
  Model = c("Ridge", "LASSO", "OLS"),
  Test_MSE = c(ridge_mse, lasso_mse, ols_mse)
)

print(results)


