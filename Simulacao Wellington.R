# ===============================
# COMPARAÇÃO: BART x RF x XGBoost x CART
# Classificação binária com dados simulados

# Correção no bloco do XGBoost, atualizão do pacote causando erro! 10/03/2026

# ===============================

# ---- 1) Pacotes ----
# Carrega os pacotes usados no script (modelos, gráficos e manipulação)
library(BART)
library(randomForest)
library(xgboost)
library(dbarts)
library(ggplot2)
library(pROC)
library(dplyr)
library(tidyr)

# ---- 2) Simular dados ----
# Carrega a função externa que gera dados binários correlacionados
source("simular dados funcao.R")

# ---- 5) Métricas ----
# Função que calcula métricas de desempenho do modelo
calc_metrics <- function(y_true, prob, pred) {
  
  # Calcula curva ROC e AUC
  roc_obj <- roc(y_true, prob)
  
  # Retorna Acurácia, AUC, Sensibilidade, Especificidade e F1-score
  data.frame(
    Acuracia = mean(y_true == pred), 
    AUC = as.numeric(auc(roc_obj)),
    Sensibilidade = sum(y_true == 1 & pred == 1) / sum(y_true == 1),
    Especificidade = sum(y_true == 0 & pred == 0) / sum(y_true == 0),
    F1_score = 2 * sum(y_true == 1 & pred == 1) / (sum(pred == 1) + sum(y_true == 1))
  )
}

set.seed(12345)
B <- 10 # Ok Número de simulações / Linha para alterar número de simulações
resultados <- vector("list", B)  # Armazena métricas por iteração
importancia_bart <- vector("list", B)
importancia_rf   <- vector("list", B)
importancia_xgb  <- vector("list", B)
importancia_cart <- vector("list", B)

for(i in 1:B){
  
  nx <- 10  #OK Número de variáveis preditoras / Linha para alterar número de variáveis
  
  # ---- Gerar dados simulados ----
  # Gera X independentes e Y correlacionado apenas com X1 e X2
  dados <- sim_bindata_class(
    n = 100, #OK tamanho da amostra / Linha para alterar o tamanho da amostra
    nx = nx,
    px = rep(0.4, nx),  #OK probabilidade das variáveis / Linha para alterar probabilidade das variáveis X
    py = 0.5,   # Linha para alterar probabilidade de Y
    x_corr = list(type = "indep"),   
    ryx = c(0.5, 0.3, rep(0, nx-2))  #OK importância das variáveis / Linha para alterar quais variáveis são relevantes
  )
  
  # Converte fatores em valores binários 0/1
  dados_num <- as.data.frame(lapply(dados, \(x) as.integer(as.character(x))))
  
  # ---- 3) Separar treino e teste ----
  idx <- sample(1:nrow(dados_num), 0.7 * nrow(dados_num))
  train <- dados_num[idx, ]
  test  <- dados_num[-idx, ]
  
  # Matriz de preditores
  Xtrain <- as.matrix(train[, -ncol(train)])
  Ytrain <- train$Y
  Xtest  <- as.matrix(test[, -ncol(test)])
  Ytest  <- test$Y
  
  # Nomear colunas para o XGBoost
  colnames(Xtrain) <- paste0("X", 1:ncol(Xtrain))
  colnames(Xtest)  <- paste0("X", 1:ncol(Xtest))
  
  # ---- 4) Ajustar modelos ----
  
  ## 4.1) BART
  # Ajusta modelo BART usando probabilidade e sparsity
  bart_fit <- pbart(
    x.train = Xtrain,
    y.train = Ytrain,
    x.test  = Xtest,
    sparse  = TRUE
  )
  prob_bart <- bart_fit$prob.test.mean
  pred_bart <- ifelse(prob_bart > 0.5, 1, 0)
  
  ## 4.2) Random Forest
  # Modelo de floresta aleatória com 300 árvores
  rf_fit <- randomForest(as.factor(Y) ~ ., data = train, ntree = 300)
  prob_rf <- predict(rf_fit, newdata = test, type = "prob")[, 2]
  pred_rf <- ifelse(prob_rf > 0.5, 1, 0)
  
  ## 4.3) XGBoost
  # Constrói objetos DMatrix
  dtrain <- xgb.DMatrix(data = Xtrain, label = Ytrain)
  dtest  <- xgb.DMatrix(data = Xtest, label = Ytest)
  
  # Ajusta modelo com função de perda logística
  params <- list(
    objective = "binary:logistic",
    eval_metric = "auc"
  )
  
  xgb_fit <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = 300,
    verbose = 0
  )
  
  prob_xgb <- predict(xgb_fit, dtest)
  pred_xgb <- ifelse(prob_xgb > 0.5, 1, 0)
  
  ## 4.4) CART
  cart_fit <- rpart(
    Y ~ ., 
    data = train,
    method = "class"
  )
  
  prob_cart <- predict(cart_fit, newdata = test, type = "prob")[,2]
  pred_cart <- ifelse(prob_cart > 0.5, 1, 0)
  
  # ---- Calcular métricas e salvar ----
  metricas <- bind_rows(
    calc_metrics(Ytest, prob_bart, pred_bart) %>% mutate(Modelo = "BART"),
    calc_metrics(Ytest, prob_rf, pred_rf)     %>% mutate(Modelo = "Random Forest"),
    calc_metrics(Ytest, prob_xgb, pred_xgb)   %>% mutate(Modelo = "XGBoost"),
    calc_metrics(Ytest, prob_cart, pred_cart) %>% mutate(Modelo = "CART")
  ) %>%
    mutate(Iteracao = i)
  
  resultados[[i]] <- metricas
  
  # ---- Importância no BART ----
  importancia_bart[[i]] <- data.frame(
    Iteracao = i,
    Modelo = "BART",
    Variavel = colnames(Xtrain),
    Importancia = bart_fit$varcount.mean
  )
  
  # ---- Random Forest ----
  imp_rf <- importance(rf_fit, type = 2)
  importancia_rf[[i]] <- data.frame(
    Iteracao = i,
    Modelo = "Random Forest",
    Variavel = rownames(imp_rf),
    Importancia = imp_rf[, 1]
  )
  
  # ---- XGBoost ----
  imp_xgb <- xgb.importance(model = xgb_fit)
  
  importancia_xgb[[i]] <- data.frame(
    Iteracao = i,
    Modelo = "XGBoost",
    Variavel = paste0("X", 1:nx)
  ) %>%
    left_join(imp_xgb[, c("Feature", "Gain")],
              by = c("Variavel" = "Feature")) %>%
    mutate(Importancia = ifelse(is.na(Gain), 0, Gain))%>%
    select(-Gain)
  
  # ---- Importância CART ----
  imp_cart <- cart_fit$variable.importance
  
  # Criar vetor completo com todas variáveis
  imp_cart_full <- setNames(
    rep(0, nx),
    paste0("X", 1:nx)
  )
  
  # Inserir importâncias reais
  imp_cart_full[names(imp_cart)] <- imp_cart
  
  # Salvar com todas variáveis
  importancia_cart[[i]] <- data.frame(
    Iteracao = i,
    Modelo = "CART",
    Variavel = names(imp_cart_full),
    Importancia = imp_cart_full
  )
}

# ---- Juntar todas métricas ----
metricas_todas <- bind_rows(resultados)

# ---- Juntar importâncias ----
importancias_todas <- bind_rows(
  bind_rows(importancia_bart) %>% transmute(Iteracao, Modelo, Variavel, Importancia = varprob),
  bind_rows(importancia_rf),
  bind_rows(importancia_xgb) %>% rename(Importancia = Gain),
  bind_rows(importancia_cart)
)



# ---- Normalizar importâncias dentro de cada iteração ----
importancias_pad <- importancias_todas %>%
  group_by(Modelo, Iteracao) %>%
  mutate(Importancia_norm = Importancia / max(Importancia, na.rm = TRUE)) %>%
  ungroup()

# ---- Exportar tabelas ----
writexl::write_xlsx(metricas_todas, "metricas.xlsx")
writexl::write_xlsx(importancias_todas, "importancias_todas.xlsx")

# ---- Resumo das métricas ----
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

# ---- Converter métricas para formato longo (para gráficos) ----
metricas_long <- metricas_todas %>%
  pivot_longer(
    cols = c(Acuracia, AUC, Sensibilidade, Especificidade, F1_score),
    names_to = "Metrica",
    values_to = "Valor"
  )

# ---- Boxplot por métrica ----
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

# ---- Boxplot facetado ----
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
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    plot.title = element_text(face = "bold", size = 14)
  )

# ---- Resumo da importância normalizada ----
resumo_importancia_norm <- importancias_pad %>%
  group_by(Modelo, Variavel) %>%
  summarise(
    Importancia_media = mean(Importancia_norm, na.rm = TRUE),
    .groups = "drop"
  )

# Padronizar nomes e ordem das variáveis
resumo_importancia_norm <- resumo_importancia_norm %>%
  mutate(
    Variavel = toupper(Variavel),
    Variavel = factor(Variavel, levels = paste0("X", 1:10), ordered = TRUE),
    Modelo = tools::toTitleCase(tolower(Modelo))
  )

# ---- Gráfico da importância média ----
ggplot(resumo_importancia_norm,
       aes(x = Variavel, y = Importancia_media, fill = Modelo)) +
  geom_col(position = "dodge") +
  labs(
    title = "Importância média normalizada das variáveis por modelo",
    x = "Variável",
    y = "Importância média (normalizada por iteração)"
  ) +
  theme_minimal(base_size = 13)

