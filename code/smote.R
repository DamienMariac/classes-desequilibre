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

# Lien de téléchargement de la base de données :  https://www.kaggle.com/mlg-ulb/creditcardfraud
# (à placer dans le dossier "code" )

df <- fread("creditcard.csv")
df$Class <- as.factor(df$Class)

# Vérifier distribution initiale
table(df$Class)

# ---------------------------
# 2. Appliquer SMOTE
# ---------------------------
X <- df[, setdiff(names(df), "Class"), with = FALSE]   # variables explicatives
y <- as.numeric(as.character(df$Class))               # cible numérique (0/1)

set.seed(123)
smote_out <- SMOTE(X, y, K = 5)   # K = nombre de voisins

# Reconstituer dataset équilibré
df_smote <- data.frame(smote_out$data)
names(df_smote)[ncol(df_smote)] <- "Class"
df_smote$Class <- as.factor(df_smote$Class)

# Vérifier distribution après SMOTE
table(df_smote$Class)

# ---------------------------
# 3. Prétraitement
# ---------------------------
X_smote <- df_smote[, setdiff(names(df_smote), "Class")]
y_smote <- df_smote$Class

scaler_smote <- preProcess(X_smote, method = c("center", "scale"))
X_smote_scaled <- predict(scaler_smote, X_smote)

# ---------------------------
# 4. Division Train/Test
# ---------------------------
set.seed(123)
trainIndex_smote <- createDataPartition(y_smote, p = 0.7, list = FALSE)
X_train_smote <- X_smote_scaled[trainIndex_smote, ]
X_test_smote  <- X_smote_scaled[-trainIndex_smote, ]
y_train_smote <- y_smote[trainIndex_smote]
y_test_smote  <- y_smote[-trainIndex_smote]

# ---------------------------
# 5. Régression logistique
# ---------------------------
logit_smote <- glm(y_train_smote ~ ., 
                   data = data.frame(X_train_smote, y_train_smote),
                   family = binomial)

# Probabilités prédites
prob_logit_smote <- predict(logit_smote, newdata = data.frame(X_test_smote), type = "response")

# Classes prédites
y_pred_class_smote <- ifelse(prob_logit_smote > 0.5, 1, 0)

# ---------------------------
# 6. Évaluation
# ---------------------------
# Courbe ROC
roc_logit_smote <- roc(as.numeric(as.character(y_test_smote)), prob_logit_smote)
plot(roc_logit_smote, col = "darkgreen", main = "ROC Logistic (SMOTE)")

# AUC
auc_value <- auc(roc_logit_smote)
print(paste("AUC Logistic SMOTE:", auc_value))

# Matrice de confusion
confusion <- confusionMatrix(factor(y_pred_class_smote), y_test_smote)
print(confusion)
