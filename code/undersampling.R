# Chargement des bibliothèques
library(data.table)
library(caret)
library(ROSE)

# Chargement des données
df <- fread("creditcard.csv")
table(df$Class)

# Séparation train/test
set.seed(123)
idx <- createDataPartition(df$Class, p = 0.8, list = FALSE)
train_set <- df[idx, ]
test_set  <- df[-idx, ]

# Sous-échantillonnage aléatoire de la classe majoritaire
n <- table(train_set$Class)[2] # nombre de fraudes

undersampled_train <- ovun.sample(Class ~ ., data = train_set,
                                  method = "under",
                                  N = 2 * n)$data

print(table(undersampled_train$Class))

# Séparation features / cible
X_train_rus <- undersampled_train[, setdiff(names(undersampled_train), "Class")]
y_train_rus <- undersampled_train$Class

X_test <- test_set[, setdiff(names(test_set), "Class"), with = FALSE]
y_test <- test_set$Class

# Mise à l'échelle (centrage-réduction)
scaler_rus <- preProcess(X_train_rus, method = c("center", "scale"))
X_train_rus_scaled <- predict(scaler_rus, X_train_rus)
X_test_rus_scaled  <- predict(scaler_rus, X_test)

# Régression logistique
logit_rus <- glm(y_train_rus ~ ., 
                 data = data.frame(X_train_rus_scaled, y_train_rus = as.factor(y_train_rus)),
                 family = binomial)

# Prédictions et évaluation
prob_rus <- predict(logit_rus, newdata = data.frame(X_test_rus_scaled), type = "response")
y_pred_rus <- ifelse(prob_rus >= 0.9, 1, 0)

# Matrice de confusion
confusion_rus <- caret::confusionMatrix(
  factor(y_pred_rus, levels = c("0","1")),
  factor(y_test,     levels = c("0","1")),
  positive = "1"
)

# Résultats
print(confusion_rus$table)
print(confusion_rus$byClass)
print(confusion_rus$overall)
