
library(data.table)
library(caret)
library(smotefamily)


df <- fread("creditcard.csv")
table(df$Class) 
trainIndex <- createDataPartition(df$Class, p = 0.8, list = FALSE)
train_set <- df[trainIndex, ]
test_set  <- df[-trainIndex, ]

X_train <- train_set[, !"Class"]
y_train <- train_set$Class

X_test <- test_set[, !"Class"]
y_test <- test_set$Class

scaler <- preProcess(X_train, method = c("center", "scale"))
X_train_scaled <- predict(scaler, X_train)
X_test_scaled  <- predict(scaler, X_test)

sm <- SMOTE(X_train_scaled, y_train, K = 5, dup_size = 0)
smote_train <- sm$data
smote_train$Class <- ifelse(smote_train$class == 1, 1, 0)
smote_train$class <- NULL

X_train_smote <- smote_train[, !"Class"]
y_train_smote <- smote_train$Class

cat("Distribution après SMOTE (TRAIN):\n")
print(table(y_train_smote))

logit_model <- glm(y_train_smote ~ ., 
                   data = data.frame(X_train_smote, y_train_smote = as.factor(y_train_smote)),
                   family = binomial)

prob <- predict(logit_model, newdata = data.frame(X_test_scaled), type = "response")
y_pred <- ifelse(prob >= 0.9, 1, 0)

confusion <- caret::confusionMatrix(
  factor(y_pred, levels = c("0","1")),
  factor(y_test, levels = c("0","1")),
  positive = "1"
)

print(confusion$table)
print(confusion$byClass)  

