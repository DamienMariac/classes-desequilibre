# ---------------------------
# Packages nécessaires
# ---------------------------
install.packages(c("data.table", "pROC", "caret", "dplyr", "smotefamily"))
library(data.table)
library(pROC)
library(caret)
library(dplyr)
library(smotefamily)

# ---------------------------
# 1. Charger les données
# ---------------------------
df <- fread("creditcard.csv")
df$Class <- as.factor(df$Class)

# Vérifier distribution initiale
table(df$Class)

# ---------------------------
# 2. Appliquer Borderline-SMOTE
# ---------------------------
X <- df[, setdiff(names(df), "Class"), with = FALSE]   # variables explicatives
y <- as.numeric(as.character(df$Class))               # cible numérique (0/1)

set.seed(123)
blsmote_out <- BLSMOTE(X, y, K = 5)   # Borderline-SMOTE avec k=5 voisins

# Reconstituer dataset équilibré
df_blsmote <- data.frame(blsmote_out$data)
names(df_blsmote)[ncol(df_blsmote)] <- "Class"
df_blsmote$Class <- as.factor(df_blsmote$Class)

# Vérifier distribution après BLSMOTE
table(df_blsmote$Class)

# ---------------------------
# 3. Prétraitement
# ---------------------------
X_blsmote <- df_blsmote[, setdiff(names(df_blsmote), "Class")]
y_blsmote <- df_blsmote$Class

scaler_blsmote <- preProcess(X_blsmote, method = c("center", "scale"))
X_blsmote_scaled <- predict(scaler_blsmote, X_blsmote)

# ---------------------------
# 4. Division Train/Test
# ---------------------------
set.seed(123)
trainIndex_blsmote <- createDataPartition(y_blsmote, p = 0.7, list = FALSE)
X_train_blsmote <- X_blsmote_scaled[trainIndex_blsmote, ]
X_test_blsmote  <- X_blsmote_scaled[-trainIndex_blsmote, ]
y_train_blsmote <- y_blsmote[trainIndex_blsmote]
y_test_blsmote  <- y_blsmote[-trainIndex_blsmote]

# ---------------------------
# 5. Régression logistique
# ---------------------------
logit_blsmote <- glm(y_train_blsmote ~ ., 
                     data = data.frame(X_train_blsmote, y_train_blsmote),
                     family = binomial)

# Probabilités prédites
prob_logit_blsmote <- predict(logit_blsmote, newdata = data.frame(X_test_blsmote), type = "response")

# Classes prédites
y_pred_class_blsmote <- ifelse(prob_logit_blsmote > 0.5, 1, 0)

# ---------------------------
# 6. Évaluation
# ---------------------------
# Courbe ROC
roc_logit_blsmote <- roc(as.numeric(as.character(y_test_blsmote)), prob_logit_blsmote)
plot(roc_logit_blsmote, col = "darkred", main = "ROC Logistic (Borderline-SMOTE)")

# AUC
auc_value <- auc(roc_logit_blsmote)
print(paste("AUC Logistic Borderline-SMOTE:", auc_value))

# Matrice de confusion
confusion <- confusionMatrix(factor(y_pred_class_blsmote), y_test_blsmote)
print(confusion)
