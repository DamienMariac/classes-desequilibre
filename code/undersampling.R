# ---------------------------
# Packages nécessaires
# ---------------------------
install.packages(c("data.table", "pROC", "caret", "dplyr"))
library(pROC)
library(data.table)
library(caret)
library(dplyr)

# ---------------------------
# 1. Charger les données
# ---------------------------

# Lien de téléchargement de la base de données :  https://www.kaggle.com/mlg-ulb/creditcardfraud
# (à placer dans le dossier "code" )

df <- fread("creditcard.csv")

X <- df[, !"Class"]
y <- df$Class

# Standardisation (important pour la régression logistique)
scaler <- preProcess(X, method = c("center", "scale"))
X_scaled <- predict(scaler, X)

# ---------------------------
# 2. Division Train/Test
# ---------------------------
set.seed(123)
trainIndex <- createDataPartition(y, p = 0.7, list = FALSE)
X_train <- X_scaled[trainIndex, ]
X_test  <- X_scaled[-trainIndex, ]
y_train <- y[trainIndex]
y_test  <- y[-trainIndex]

# ---------------------------
# 3. Logistic Regression (dataset complet)
# ---------------------------
logit <- glm(y_train ~ ., 
             data = data.frame(X_train, y_train = as.factor(y_train)),
             family = binomial)

# Probabilités prédites
prob_logit <- predict(logit, newdata = data.frame(X_test), type = "response")

# Conversion en classes (seuil = 0.5 par défaut)
y_pred_class <- ifelse(prob_logit > 0.5, 1, 0)

# Courbe ROC
roc_logit <- pROC::roc(y_test, prob_logit)
plot(roc_logit, col = "blue", main = "ROC Logistic (Full Data)")

# Matrice de confusion
caret::confusionMatrix(factor(y_pred_class), factor(y_test))

# AUC
pROC::auc(roc_logit)


# ---------------------------
# 4. Random Undersampling
# ---------------------------
fraud_idx <- which(y == 1)
normal_idx <- which(y == 0)

set.seed(123)
random_normal_idx <- sample(normal_idx, length(fraud_idx))

under_idx <- c(fraud_idx, random_normal_idx)
df_under <- df[under_idx, ]

X_under <- df_under[, !"Class"]
y_under <- df_under$Class

# Re-standardisation
scaler_under <- preProcess(X_under, method = c("center", "scale"))
X_under_scaled <- predict(scaler_under, X_under)

set.seed(123)
trainIndex_under <- createDataPartition(y_under, p = 0.7, list = FALSE)
X_train_under <- X_under_scaled[trainIndex_under, ]
X_test_under  <- X_under_scaled[-trainIndex_under, ]
y_train_under <- y_under[trainIndex_under]
y_test_under  <- y_under[-trainIndex_under]

# ---------------------------
# 5. Logistic Regression (undersampled data)
# ---------------------------
logit_under <- glm(y_train_under ~ ., 
                   data = data.frame(X_train_under, y_train_under = as.factor(y_train_under)),
                   family = binomial)

# Probabilités prédites
prob_logit_under <- predict(logit_under, newdata = data.frame(X_test_under), type = "response")

# Conversion en classes (seuil = 0.5)
y_pred_class_under <- ifelse(prob_logit_under > 0.5, 1, 0)

# Courbe ROC
roc_logit_under <- pROC::roc(y_test_under, prob_logit_under)
plot(roc_logit_under, col = "green", main = "ROC Logistic (Undersampled)")

# Matrice de confusion
caret::confusionMatrix(factor(y_pred_class_under), factor(y_test_under))

# AUC
pROC::auc(roc_logit_under)

