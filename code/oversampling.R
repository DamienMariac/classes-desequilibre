library(data.table)
library(caret)
library(ROSE)     

df <- fread("creditcard.csv")
table(df$Class)

idx <- createDataPartition(df$Class, p = 0.8, list = FALSE)
train_set <- df[idx, ]
test_set  <- df[-idx, ]

oversampled_train <- ovun.sample(Class ~ ., data = train_set, method = "over",
                                 N = 2 * table(train_set$Class)[1])$data
print(table(oversampled_train$Class))

X_train <- oversampled_train[, setdiff(names(oversampled_train), "Class")]
y_train <- oversampled_train$Class


X_test <- test_set[, !"Class"]
y_test <- test_set$Class

scaler <- preProcess(X_train, method = c("center", "scale"))
X_train_scaled <- predict(scaler, X_train)
X_test_scaled  <- predict(scaler, X_test)

logit_model <- glm(y_train ~ ., 
                   data = data.frame(X_train_scaled, y_train = as.factor(y_train)),
                   family = binomial)

prob <- predict(logit_model, newdata = data.frame(X_test_scaled), type = "response")

y_pred <- ifelse(prob >= 0.9, 1, 0)

confusion <- caret::confusionMatrix(
  factor(y_pred, levels = c("0","1")),
  factor(y_test, levels = c("0","1")),
  positive = "1"
)

print(confusion$table)

print(confusion$byClass)   # Sensitivity, Specificity, Precision, Recall, F1, etc.
print(confusion$overall)   # Accuracy

