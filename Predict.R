############################################
#### LINEAR REGRESSION
############################################
# Fit
model <- lm(y ~ x1 + x2, data = train_data)
# Predict
predictions <- predict(model, newdata = test_data)


############################################
#### RIDGE REGRESSION
############################################
# Fit (glmnet)
library(glmnet)
x_train <- model.matrix(y ~ . + 0, data = train_data)
model <- glmnet(x_train, train_data$y, alpha = 0, lambda = 0.1)
# Predict
x_test <- model.matrix(~ . + 0, test_data)
predictions <- predict(model, newx = x_test)
# CV for lambda selection
cv_model <- cv.glmnet(x_train, train_data$y, alpha = 0)
best_lambda <- cv_model$lambda.min
model <- glmnet(x_train, train_data$y, alpha = 0, lambda = best_lambda)


############################################
#### LASSO
############################################
# Fit (glmnet)
library(glmnet)
x_train <- model.matrix(y ~ . + 0, data = train_data)
model <- glmnet(x_train, train_data$y, alpha = 1, lambda = 0.1)
# Predict
x_test <- model.matrix(~ . + 0, test_data)
predictions <- predict(model, newx = x_test)
# CV for lambda selection
cv_model <- cv.glmnet(x_train, train_data$y, alpha = 1)
best_lambda <- cv_model$lambda.min
model <- glmnet(x_train, train_data$y, alpha = 1, lambda = best_lambda)


############################################
#### LOGISTIC REGRESSION
############################################
# Fit
model <- glm(y ~ x1 + x2, data = train_data, family = "binomial")
# Predict classes
class_pred <- predict(model, newdata = test_data, type = "response") > 0.5
# Predict probabilities (for ROC/AUC)
prob_pred <- predict(model, newdata = test_data, type = "response")


############################################
#### K-NEAREST NEIGHBORS
############################################
# Fit & predict in one step (class package)
library(class)
predictions <- knn(train = train_data[, -1], test = test_data[, -1], cl = train_data$y, k = 5)
# For probabilities
predictions <- knn(train = train_data[, -1], test = test_data[, -1], cl = train_data$y, k = 5, prob = TRUE)
probs <- attr(predictions, "prob")


############################################
#### LINEAR DISCRIMINANT ANALYSIS
############################################
# Fit (MASS package)
library(MASS)
model <- lda(y ~ ., data = train_data)
# Predict
predictions <- predict(model, newdata = test_data)
# Access classes and probabilities
classes <- predictions$class
probs <- predictions$posterior


############################################
#### QUADRATIC DISCRIMINANT ANALYSIS
############################################
# Fit (MASS package)
library(MASS)
model <- qda(y ~ ., data = train_data)
# Predict
predictions <- predict(model, newdata = test_data)
# Access classes and probabilities
classes <- predictions$class
probs <- predictions$posterior


############################################
#### DECISION TREES (tree package)
############################################
# Fit (tree package)
library(tree)
model <- tree(y ~ ., data = train_data)
# Predict classes (for classification)
class_pred <- predict(model, newdata = test_data, type = "class")
# Predict probabilities (for classification)
prob_pred <- predict(model, newdata = test_data)
# Predict values (for regression)
reg_pred <- predict(model, newdata = test_data)
# Pruning
cv_tree <- cv.tree(model, FUN = prune.misclass)  # For classification
# cv_tree <- cv.tree(model)  # For regression
opt_size <- cv_tree$size[which.min(cv_tree$dev)]
pruned_model <- prune.misclass(model, best = opt_size)  # For classification
# pruned_model <- prune.tree(model, best = opt_size)  # For regression


############################################
#### RANDOM FORESTS
############################################
# Fit (randomForest package)
library(randomForest)
model <- randomForest(y ~ ., data = train_data, ntree = 500)
# Predict
predictions <- predict(model, newdata = test_data)
# Predict probabilities
prob_pred <- predict(model, newdata = test_data, type = "prob")
# Variable importance
importance(model)
varImpPlot(model)


############################################
#### BAGGING
############################################
# Fit (randomForest with all features)
library(randomForest)
p <- ncol(train_data) - 1  # number of predictors
model <- randomForest(y ~ ., data = train_data, mtry = p)
# Predict
predictions <- predict(model, newdata = test_data)


############################################
#### GRADIENT BOOSTING
############################################
# Fit (gbm package)
library(gbm)
model <- gbm(y ~ ., data = train_data, distribution = "bernoulli", n.trees = 100)
# Predict
predictions <- predict(model, newdata = test_data, n.trees = 100, type = "response")
# For regression
model <- gbm(y ~ ., data = train_data, distribution = "gaussian", n.trees = 100)


############################################
#### SUPPORT VECTOR MACHINES (LINEAR)
############################################
# Fit (e1071 package)
library(e1071)
model <- svm(y ~ ., data = train_data, kernel = "linear")
# Predict classes
predictions <- predict(model, newdata = test_data)
# Fit with probability for ROC
model <- svm(y ~ ., data = train_data, kernel = "linear", probability = TRUE)
# Get probabilities
prob_pred <- predict(model, newdata = test_data, probability = TRUE)
probs <- attr(prob_pred, "probabilities")


############################################
#### SUPPORT VECTOR MACHINES (RADIAL)
############################################
# Fit (e1071 package)
library(e1071)
model <- svm(y ~ ., data = train_data, kernel = "radial", gamma = 0.5)
# Predict
predictions <- predict(model, newdata = test_data)
# For probabilities
model <- svm(y ~ ., data = train_data, kernel = "radial", probability = TRUE)
prob_pred <- predict(model, newdata = test_data, probability = TRUE)
probs <- attr(prob_pred, "probabilities")


############################################
#### NEURAL NETWORKS
############################################
# Scale data first
library(nnet)
train_scaled <- scale(train_data[, -1])
test_scaled <- scale(test_data[, -1], 
                    center = attr(train_scaled, "scaled:center"),
                    scale = attr(train_scaled, "scaled:scale"))

# Fit (nnet package)
model <- nnet(y ~ ., data = data.frame(y = train_data$y, train_scaled), size = 5, decay = 0.01)
# Predict classes
predictions <- predict(model, newdata = data.frame(test_scaled), type = "class")
# Predict probabilities
prob_pred <- predict(model, newdata = data.frame(test_scaled), type = "raw")


############################################
#### COX PROPORTIONAL HAZARDS MODEL
############################################
# Fit (survival package)
library(survival)
model <- coxph(Surv(time, status) ~ x1 + x2, data = train_data)
# Predict risk scores
predictions <- predict(model, newdata = test_data, type = "risk")
# Predict survival curves
surv_curves <- survfit(model, newdata = test_data)


############################################
#### PRINCIPAL COMPONENT ANALYSIS
############################################
# Fit
pca <- prcomp(train_data[, -1], center = TRUE, scale. = TRUE)
# Project data to PC space
pc_scores <- predict(pca, newdata = test_data[, -1])
# Variance explained
var_explained <- pca$sdev^2 / sum(pca$sdev^2)
# Reconstruct data using first 2 components
reconstructed <- pc_scores[, 1:2] %*% t(pca$rotation[, 1:2])


############################################
#### MODEL EVALUATION
############################################
# Confusion Matrix (caret)
library(caret)
confusionMatrix(predictions, test_data$y)

# ROC curve and AUC (pROC)
library(pROC)
roc_obj <- roc(test_data$y, prob_pred)
auc_value <- auc(roc_obj)
plot(roc_obj)

# Mean Squared Error (for regression)
mse <- mean((predictions - test_data$y)^2)
rmse <- sqrt(mse)

# R-squared (for regression)
r_squared <- cor(predictions, test_data$y)^2


############################################
#### CARET IMPLEMENTATIONS
############################################
library(caret)

# Create trainControl object for resampling
# For classification
ctrl_class <- trainControl(
  method = "cv",                # k-fold cross-validation
  number = 5,                   # number of folds
  classProbs = TRUE,            # compute class probabilities
  summaryFunction = twoClassSummary,  # ROC, Sens, Spec metrics
  savePredictions = "final"     # save final predictions
)

# For regression
ctrl_reg <- trainControl(
  method = "cv",                # k-fold cross-validation
  number = 5                    # number of folds
)

# Linear Regression (caret)
lm_model <- train(
  y ~ .,
  data = train_data,
  method = "lm",
  trControl = ctrl_reg
)
lm_preds <- predict(lm_model, newdata = test_data)

# Ridge Regression (caret)
ridge_model <- train(
  y ~ .,
  data = train_data,
  method = "glmnet",
  tuneGrid = expand.grid(alpha = 0, 
                        lambda = seq(0.0001, 1, length = 20)),
  trControl = ctrl_reg
)
ridge_preds <- predict(ridge_model, newdata = test_data)

# LASSO (caret)
lasso_model <- train(
  y ~ .,
  data = train_data,
  method = "glmnet",
  tuneGrid = expand.grid(alpha = 1, 
                        lambda = seq(0.0001, 1, length = 20)),
  trControl = ctrl_reg
)
lasso_preds <- predict(lasso_model, newdata = test_data)

# Elastic Net (caret)
enet_model <- train(
  y ~ .,
  data = train_data,
  method = "glmnet",
  tuneGrid = expand.grid(alpha = seq(0, 1, length = 5), 
                        lambda = seq(0.0001, 1, length = 5)),
  trControl = ctrl_reg
)
enet_preds <- predict(enet_model, newdata = test_data)

# Logistic Regression (caret)
logreg_model <- train(
  y ~ .,
  data = train_data,
  method = "glm",
  family = "binomial",
  trControl = ctrl_class,
  metric = "ROC"
)
logreg_preds <- predict(logreg_model, newdata = test_data)
logreg_probs <- predict(logreg_model, newdata = test_data, type = "prob")

# K-Nearest Neighbors (caret)
knn_model <- train(
  y ~ .,
  data = train_data,
  method = "knn",
  tuneGrid = data.frame(k = seq(1, 20, by = 2)),
  trControl = ctrl_class,
  metric = "ROC"
)
knn_preds <- predict(knn_model, newdata = test_data)
knn_probs <- predict(knn_model, newdata = test_data, type = "prob")

# Linear Discriminant Analysis (caret)
lda_model <- train(
  y ~ .,
  data = train_data,
  method = "lda",
  trControl = ctrl_class,
  metric = "ROC"
)
lda_preds <- predict(lda_model, newdata = test_data)
lda_probs <- predict(lda_model, newdata = test_data, type = "prob")

# Quadratic Discriminant Analysis (caret)
qda_model <- train(
  y ~ .,
  data = train_data,
  method = "qda",
  trControl = ctrl_class,
  metric = "ROC"
)
qda_preds <- predict(qda_model, newdata = test_data)
qda_probs <- predict(qda_model, newdata = test_data, type = "prob")

# Decision Trees (caret)
tree_model <- train(
  y ~ .,
  data = train_data,
  method = "rpart",  # uses rpart not tree
  tuneLength = 10,
  trControl = ctrl_class,
  metric = "ROC"
)
tree_preds <- predict(tree_model, newdata = test_data)
tree_probs <- predict(tree_model, newdata = test_data, type = "prob")

# Random Forest (caret)
rf_model <- train(
  y ~ .,
  data = train_data,
  method = "rf",
  tuneLength = 5,
  trControl = ctrl_class,
  metric = "ROC"
)
rf_preds <- predict(rf_model, newdata = test_data)
rf_probs <- predict(rf_model, newdata = test_data, type = "prob")

# Gradient Boosting (caret)
gbm_model <- train(
  y ~ .,
  data = train_data,
  method = "gbm",
  tuneLength = 5,
  trControl = ctrl_class,
  metric = "ROC",
  verbose = FALSE
)
gbm_preds <- predict(gbm_model, newdata = test_data)
gbm_probs <- predict(gbm_model, newdata = test_data, type = "prob")

# Support Vector Machine - Linear (caret)
svm_linear_model <- train(
  y ~ .,
  data = train_data,
  method = "svmLinear",
  tuneLength = 5,
  trControl = ctrl_class,
  metric = "ROC"
)
svm_linear_preds <- predict(svm_linear_model, newdata = test_data)
svm_linear_probs <- predict(svm_linear_model, newdata = test_data, type = "prob")

# Support Vector Machine - Radial (caret)
svm_radial_model <- train(
  y ~ .,
  data = train_data,
  method = "svmRadial",
  tuneLength = 5,
  trControl = ctrl_class,
  metric = "ROC"
)
svm_radial_preds <- predict(svm_radial_model, newdata = test_data)
svm_radial_probs <- predict(svm_radial_model, newdata = test_data, type = "prob")

# Neural Network (caret)
nnet_model <- train(
  y ~ .,
  data = train_data,
  method = "nnet",
  tuneLength = 5,
  trControl = ctrl_class,
  metric = "ROC",
  trace = FALSE,
  maxit = 100
)
nnet_preds <- predict(nnet_model, newdata = test_data)
nnet_probs <- predict(nnet_model, newdata = test_data, type = "prob")

# Cox Proportional Hazards (caret)
# Requires survival data with Surv object
cox_model <- train(
  x = train_data[, -c(1, 2)],  # Remove time and status columns
  y = Surv(train_data$time, train_data$status),
  method = "coxph",
  trControl = trainControl(method = "cv", number = 5)
)
cox_preds <- predict(cox_model, newdata = test_data)

# Principal Component Analysis (caret)
# For dimensionality reduction before modeling
preProc <- preProcess(train_data[, -1], method = "pca", pcaComp = 2)
train_pca <- predict(preProc, train_data[, -1])
test_pca <- predict(preProc, test_data[, -1])

# Model Comparison
model_list <- list(
  LR = lm_model,
  Ridge = ridge_model, 
  LASSO = lasso_model,
  LogReg = logreg_model,
  KNN = knn_model,
  LDA = lda_model,
  QDA = qda_model,
  Tree = tree_model,
  RF = rf_model,
  GBM = gbm_model,
  SVM_Linear = svm_linear_model,
  SVM_Radial = svm_radial_model,
  NNet = nnet_model
)

# For classification
resamples_obj <- resamples(model_list)
summary(resamples_obj)
dotplot(resamples_obj, metric = "ROC")

# For regression (replace model_list with regression models)
# dotplot(resamples_obj, metric = "RMSE")
