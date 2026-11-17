# ===============================
# COMPARAÇÃO: BART x RF x XGBoost x Regressão Logística
# Classificação binária com dados simulados
# ===============================

# ---- 1) Pacotes ----
# install.packages(c("BART", "randomForest", "xgboost", "ggplot2", "pROC", "dplyr", "tidyr"))
library(BART)
library(randomForest)
library(xgboost)
library(ggplot2)
library(pROC)
library(dplyr)
library(tidyr)

# ---- 2) Simular dados ----
source("simular dados funcao.R")

# ---- 5) Métricas ----
calc_metrics <- function(y_true, prob, pred) {
  roc_obj <- roc(y_true, prob)
  data.frame(
    Acuracia = mean(y_true == pred),
    AUC = as.numeric(auc(roc_obj)),
    Sensibilidade = sum(y_true == 1 & pred == 1) / sum(y_true == 1),
    Especificidade = sum(y_true == 0 & pred == 0) / sum(y_true == 0),
    F1_score = 2 * sum(y_true == 1 & pred == 1) / (sum(pred == 1) + sum(y_true == 1))
  )
}

set.seed(12345)
B <- 10  #numero de simulações
resultados <- vector("list", B)  # lista para armazenar por iteração
importancia_bart <- vector("list", B)
importancia_rf   <- vector("list", B)
importancia_xgb  <- vector("list", B)
importancia_glm <- vector("list", B)
for(i in 1:B){
  nx <- 10


  dados <- sim_bindata_class(
    n = 2000,
    nx = nx,
    px = rep(0.4, nx),   # P(Xj=1) = 0.3 para todos
    py = 0.5,            # P(Y=1) = 0.5
    x_corr = list(type = "indep"),   # X independentes
    ryx = c(0.4, 0.4, rep(0, nx-2))  # Y correlacionado só com X1 e X2
  )

  # Converter fatores em numéricos (0/1)
  dados_num <- as.data.frame(lapply(dados, \(x) as.integer(as.character(x))))

  # ---- 3) Treino / teste ----
  idx <- sample(1:nrow(dados_num), 0.7 * nrow(dados_num))
  train <- dados_num[idx, ]
  test  <- dados_num[-idx, ]

  Xtrain <- as.matrix(train[, -ncol(train)])
  Ytrain <- train$Y
  Xtest  <- as.matrix(test[, -ncol(test)])
  Ytest  <- test$Y

  # Dar nomes às colunas (necessário para XGBoost)
  colnames(Xtrain) <- paste0("X", 1:ncol(Xtrain))
  colnames(Xtest)  <- paste0("X", 1:ncol(Xtest))

  # ---- 4) Modelos ----

  ## 4.1) BART
  bart_fit <- pbart(
    x.train = Xtrain,
    y.train = Ytrain,
    x.test  = Xtest,
    sparse  = TRUE
  )
  prob_bart <- bart_fit$prob.test.mean
  pred_bart <- ifelse(prob_bart > 0.5, 1, 0)

  ## 4.2) Random Forest
  rf_fit <- randomForest(as.factor(Y) ~ ., data = train, ntree = 300)
  prob_rf <- predict(rf_fit, newdata = test, type = "prob")[, 2]
  pred_rf <- ifelse(prob_rf > 0.5, 1, 0)

  ## 4.3) XGBoost
  dtrain <- xgb.DMatrix(data = Xtrain, label = Ytrain)
  dtest  <- xgb.DMatrix(data = Xtest, label = Ytest)
  xgb_fit <- xgboost(
    data = dtrain,
    objective = "binary:logistic",
    nrounds = 300,
    verbose = 0
  )
  prob_xgb <- predict(xgb_fit, dtest)
  pred_xgb <- ifelse(prob_xgb > 0.5, 1, 0)

  ## 4.4) Regressão Logística
  glm_fit <- glm(Y ~ ., data = train, family = binomial)
  prob_glm <- predict(glm_fit, newdata = test, type = "response")
  pred_glm <- ifelse(prob_glm > 0.5, 1, 0)


  metricas <- bind_rows(
    calc_metrics(Ytest, prob_bart, pred_bart) %>% mutate(Modelo = "BART"),
    calc_metrics(Ytest, prob_rf, pred_rf)     %>% mutate(Modelo = "Random Forest"),
    calc_metrics(Ytest, prob_xgb, pred_xgb)   %>% mutate(Modelo = "XGBoost"),
    calc_metrics(Ytest, prob_glm, pred_glm)   %>% mutate(Modelo = "Regressão Logística")
  ) %>%
    mutate(Iteracao = i)

  resultados[[i]] <- metricas
  
  importancia_bart[[i]] <- data.frame(
    Iteracao = i,
    Modelo = "BART",
    Variavel = colnames(Xtrain),
    varprob  = bart_fit$varprob.mean,
    varcount = bart_fit$varcount.mean
  )
  
  # ---- Random Forest ----
  imp_rf <- importance(rf_fit, type = 2)  # tipo = 2 → Gini
  importancia_rf[[i]] <- data.frame(
    Iteracao = i,
    Modelo = "Random Forest",
    Variavel = rownames(imp_rf),
    Importancia = imp_rf[, 1]
  )
  
  # ---- XGBoost ----
  imp_xgb <- xgb.importance(model = xgb_fit)
  # garantir que todas as variáveis apareçam
  importancia_xgb[[i]] <- data.frame(
    Iteracao = i,
    Modelo = "XGBoost",
    Variavel = paste0("X", 1:nx)
  ) %>%
    left_join(imp_xgb[, c("Feature", "Gain")],
              by = c("Variavel" = "Feature")) %>%
    mutate(Gain = ifelse(is.na(Gain), 0, Gain))
  
  coefs <- coef(glm_fit)
  coefs <- coefs[names(coefs) != "(Intercept)"]
  
  importancia_glm[[i]] <- data.frame(
    Iteracao   = i,
    Modelo     = "GLM",
    Variavel   = names(coefs),
    Importancia = abs(coefs)
  )
  

}
metricas_todas <- bind_rows(resultados)

importancias_todas <- bind_rows(
  bind_rows(importancia_bart) %>%
    transmute(Iteracao, Modelo, Variavel, Importancia = varprob),
  bind_rows(importancia_rf),
  bind_rows(importancia_xgb) %>% rename(Importancia = Gain),
  importancia_glm
)

importancias_pad <- importancias_todas %>%
  group_by(Modelo, Iteracao) %>%
  mutate(
    Importancia_norm = Importancia / max(Importancia, na.rm = TRUE)
  ) %>%
  ungroup()



writexl::write_xlsx(metricas_todas, "metricas.xlsx")
writexl::write_xlsx(metricas_todas, "importancias_todas.xlsx")

resumo_metricas <- metricas_todas %>%
  group_by(Modelo) %>%
  summarise(
    Acuracia_med  = mean(Acuracia),
    Acuracia_sd   = sd(Acuracia),
    AUC_med       = mean(AUC),
    AUC_sd        = sd(AUC),
    Sens_med      = mean(Sensibilidade),
    Sens_sd       = sd(Sensibilidade),
    Espec_med     = mean(Especificidade),
    Espec_sd      = sd(Especificidade),
    F1_med        = mean(F1_score),
    F1_sd         = sd(F1_score),
    .groups = "drop"
  )

resumo_metricas

# --- Converter para formato longo ---
metricas_long <- metricas_todas %>%
  pivot_longer(
    cols = c(Acuracia, AUC, Sensibilidade, Especificidade, F1_score),
    names_to = "Metrica",
    values_to = "Valor"
  )

# --- Boxplot por métrica, agrupando modelos ---
ggplot(metricas_long, aes(x = Metrica, y = Valor, fill = Modelo)) +
  geom_boxplot(outlier.shape = 21, alpha = 0.8) +
  theme_minimal(base_size = 13) +
  labs(
    title = "Distribuição das Métricas por Modelo",
    x = "Métrica de Avaliação",
    y = "Valor",
    fill = "Modelo"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title = element_text(face = "bold", size = 14)
  )


ggplot(metricas_long, aes(x = Modelo, y = Valor, fill = Modelo)) +
  geom_boxplot(alpha = 0.8) +
  facet_wrap(~ Metrica, scales = "free_y") +
  theme_minimal(base_size = 13) +
  labs(
    title = "Comparação dos Modelos por Métrica",
    x = NULL,
    y = "Valor"
  ) +
  theme(
    axis.text.x = element_blank(),   # remove textos do eixo x
    axis.ticks.x = element_blank(),  # remove marcas (opcional)
    plot.title = element_text(face = "bold", size = 14)
  )

# Resumo: média da importância normalizada
resumo_importancia_norm <- importancias_pad %>%
  group_by(Modelo, Variavel) %>%
  summarise(
    Importancia_media = mean(Importancia_norm, na.rm = TRUE),
    .groups = "drop"
  )

resumo_importancia_norm <- resumo_importancia_norm %>%
  mutate(
    # 1️⃣ Padronizar variáveis para maiúsculas
    Variavel = toupper(Variavel),
    
    # 2️⃣ Garantir a ordem correta (X1 → X10)
    Variavel = factor(
      Variavel,
      levels = paste0("X", 1:10),
      ordered = TRUE
    ),
    
    # 3️⃣ (Opcional) Padronizar nome do modelo também
    Modelo = tools::toTitleCase(tolower(Modelo))
  )
ggplot(resumo_importancia_norm,
       aes(x = Variavel,
           y = Importancia_media,
           fill = Modelo)) +
  geom_col(position = "dodge") +
  labs(
    title = "Importância média normalizada das variáveis por modelo",
    x = "Variável",
    y = "Importância média (normalizada por iteração)"
  ) +
  theme_minimal(base_size = 13)
