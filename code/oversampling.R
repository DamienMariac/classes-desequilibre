# ---------------------------
# Packages nécessaires
# ---------------------------
install.packages(c("data.table", "pROC", "caret", "dplyr", "ROSE"))
library(pROC)
library(data.table)
library(caret)
library(dplyr)
library(ROSE)   # pour ovun.sample()

# ---------------------------
# 1. Charger les données
# ---------------------------

# Lien de téléchargement de la base de données :  https://www.kaggle.com/mlg-ulb/creditcardfraud
# (à placer à la racine du projet)

df <- fread("creditcard.csv")

# Vérifier la distribution initiale
table(df$Class)

# ---------------------------
# 2. Random Oversampling
# ---------------------------
set.seed(123)
oversample_data <- ovun.sample(Class ~ ., data = df, method = "over", 
                               N = 2 * table(df$Class)[1])$data

# Vérifier distribution après oversampling
table(oversample_data$Class)

X_over <- oversample_data[, setdiff(names(oversample_data), "Class")]
y_over <- oversample_data$Class

# Standardisation
scaler_over <- preProcess(X_over, method = c("center", "scale"))
X_over_scaled <- predict(scaler_over, X_over)

# ---------------------------
# 3. Division Train/Test
# ---------------------------
set.seed(123)
trainIndex_over <- createDataPartition(y_over, p = 0.7, list = FALSE)
X_train_over <- X_over_scaled[trainIndex_over, ]
X_test_over  <- X_over_scaled[-trainIndex_over, ]
y_train_over <- y_over[trainIndex_over]
y_test_over  <- y_over[-trainIndex_over]

# ---------------------------
# 4. Régression logistique
# ---------------------------
logit_over <- glm(y_train_over ~ ., 
                  data = data.frame(X_train_over, y_train_over = as.factor(y_train_over)),
                  family = binomial)

# Probabilités prédites
prob_logit_over <- predict(logit_over, newdata = data.frame(X_test_over), type = "response")

# Classes prédites
y_pred_class_over <- ifelse(prob_logit_over > 0.5, 1, 0)

# ---------------------------
# 5. Évaluation
# ---------------------------
# Courbe ROC
roc_logit_over <- pROC::roc(y_test_over, prob_logit_over)
plot(roc_logit_over, col = "orange", main = "ROC Logistic (Oversampled)")

# AUC
auc_value <- pROC::auc(roc_logit_over)
print(paste("AUC Logistic Oversampled:", auc_value))

# Matrice de confusion
confusion <- caret::confusionMatrix(factor(y_pred_class_over), factor(y_test_over))
print(confusion)
