# Project 3 reproducible analysis
required <- c("completejourney", "dplyr", "ggplot2", "cluster", "pROC")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) install.packages(missing)
library(dplyr); library(ggplot2); library(cluster)
set.seed(42)
dir.create("outputs", showWarnings = FALSE); dir.create("figures", showWarnings = FALSE)

# Large-data import. get_transactions downloads the full transaction table when needed.
transactions <- completejourney::get_transactions(verbose = TRUE)
products <- completejourney::products
demographics <- completejourney::demographics
transactions$transaction_timestamp <- as.POSIXct(transactions$transaction_timestamp, tz = "UTC")
cutoff <- as.POSIXct("2017-10-01", tz = "UTC")
feature_tx <- filter(transactions, transaction_timestamp < cutoff)
holdout_tx <- filter(transactions, transaction_timestamp >= cutoff)

# Customer-level feature construction.
features <- feature_tx %>% group_by(household_id) %>%
  summarise(last_purchase = max(transaction_timestamp), frequency = n_distinct(basket_id),
            monetary = sum(sales_value), items = sum(quantity), distinct_products = n_distinct(product_id), .groups = "drop") %>%
  mutate(avg_basket = monetary / frequency, recency = as.numeric(difftime(cutoff, last_purchase, units = "days")))
category_features <- feature_tx %>% left_join(select(products, product_id, department, product_category), by = "product_id") %>%
  group_by(household_id) %>% summarise(category_count = n_distinct(department), product_category_count = n_distinct(product_category), .groups = "drop")
features <- left_join(features, category_features, by = "household_id")
retained <- holdout_tx %>% count(household_id, name = "retained") %>% mutate(retained = 1L)
customers <- demographics %>% left_join(features, by = "household_id") %>% left_join(retained, by = "household_id") %>% mutate(retained = coalesce(retained, 0L), family_size = as.numeric(household_size))
numeric_features <- c("recency", "frequency", "monetary", "avg_basket", "items", "distinct_products", "category_count", "product_category_count", "family_size")
customers[numeric_features] <- lapply(customers[numeric_features], function(x) { x[is.na(x)] <- median(x, na.rm = TRUE); x })

# PCA: standardize behavior variables and inspect explained variance/loadings.
scaled <- scale(customers[numeric_features])
pca <- prcomp(scaled)
pca_variance <- data.frame(component = paste0("PC", seq_along(pca$sdev)), explained_variance = pca$sdev^2 / sum(pca$sdev^2))
pca_variance$cumulative_variance <- cumsum(pca_variance$explained_variance)
pca_loadings <- data.frame(feature = rownames(pca$rotation), pca$rotation, row.names = NULL)
write.csv(pca_variance, "outputs/pca_variance.csv", row.names = FALSE)
write.csv(pca_loadings, "outputs/pca_loadings.csv", row.names = FALSE)

# K-means diagnostics on the first five PCA scores.
scores <- pca$x[, 1:5]
diagnostics <- do.call(rbind, lapply(2:6, function(k) {
  fit <- kmeans(scores, centers = k, nstart = 30)
  sil <- silhouette(fit$cluster, dist(scores))
  data.frame(k = k, inertia = fit$tot.withinss, silhouette = mean(sil[, 3]), min_cluster = min(table(fit$cluster)))
}))
write.csv(diagnostics, "outputs/k_diagnostics.csv", row.names = FALSE)
# k = 2 has the best silhouette, but k = 4 is selected for the final business
# segmentation because it creates four stable, actionable customer roles.
best_k <- 4
fit <- kmeans(scores, centers = best_k, nstart = 50)
customers$cluster <- factor(fit$cluster)
customers <- cbind(customers, as.data.frame(pca$x[, 1:5]))
names(customers)[(ncol(customers)-4):ncol(customers)] <- paste0("PC", 1:5)

# Interpret the clusters using original variables, not only PCA scores.
 total_households <- nrow(customers)
 profile <- customers %>% group_by(cluster) %>% summarise(customers = n(), recency = median(recency), frequency = median(frequency), monetary = median(monetary), avg_basket = median(avg_basket), return_rate = mean(retained), pct_customers = n()/total_households, .groups = "drop")
high_value_cluster <- profile$cluster[which.max(profile$monetary)]
at_risk_cluster <- profile$cluster[which.max(profile$recency)]
low_value_cluster <- profile$cluster[which.min(profile$monetary)]
profile$segment <- ifelse(profile$cluster == high_value_cluster, "High-Value Loyal Customers", ifelse(profile$cluster == at_risk_cluster, "Recently Lapsed / At-Risk Customers", ifelse(profile$cluster == low_value_cluster, "Low-Value Low-Touch Customers", "Middle-Value Growth Customers")))
customers <- left_join(customers, select(profile, cluster, segment), by = "cluster")
write.csv(profile, "outputs/segmentation_profile.csv", row.names = FALSE)

# Logistic regression evaluated on a held-out household sample.
negative_id <- which(customers$retained == 0)
positive_id <- which(customers$retained == 1)
test_negative <- if (length(negative_id) > 1) sample(negative_id, length(negative_id) - 1) else integer(0)
test_positive <- sample(positive_id, floor(.25 * length(positive_id)))
test_id <- c(test_negative, test_positive)
train_id <- setdiff(seq_len(nrow(customers)), test_id)
formula <- retained ~ recency + frequency + monetary + avg_basket + items + distinct_products + category_count + product_category_count + family_size
model <- glm(formula, data = customers[train_id, ], family = binomial())
test <- customers[test_id, ]
prob <- predict(model, newdata = test, type = "response")
predicted <- as.integer(prob >= .5)
cm <- table(Actual = test$retained, Predicted = predicted)
write.csv(as.data.frame.matrix(cm), "outputs/confusion_matrix.csv")
auc <- as.numeric(pROC::auc(test$retained, prob))
sensitivity <- sum(predicted == 1 & test$retained == 1) / sum(test$retained == 1)
specificity <- sum(predicted == 0 & test$retained == 0) / sum(test$retained == 0)
precision <- sum(predicted == 1 & test$retained == 1) / sum(predicted == 1)
write.csv(data.frame(model = "Original behavior variables", AUC = auc, sensitivity = sensitivity, specificity = specificity, precision = precision, cutoff = .5), "outputs/model_metrics.csv", row.names = FALSE)
customers$predicted_return_probability <- predict(model, newdata = customers, type = "response")

# Compare a PCA-score model and document multicollinearity in the original features.
pca_model <- glm(retained ~ PC1 + PC2 + PC3 + PC4 + PC5, data = customers[train_id, ], family = binomial())
pca_prob <- predict(pca_model, newdata = test, type = "response")
comparison <- data.frame(model = c("Original behavior variables", "First five PCA scores"), AUC = c(auc, as.numeric(pROC::auc(test$retained, pca_prob))))
write.csv(comparison, "outputs/model_comparison.csv", row.names = FALSE)
correlations <- cor(customers[numeric_features], use = "pairwise.complete.obs")
correlations[lower.tri(correlations, diag = TRUE)] <- NA
max_pair <- which(abs(correlations) == max(abs(correlations), na.rm = TRUE), arr.ind = TRUE)[1, ]
write.csv(data.frame(feature_1 = rownames(correlations)[max_pair[1]], feature_2 = colnames(correlations)[max_pair[2]], correlation = correlations[max_pair[1], max_pair[2]]), "outputs/max_correlation.csv", row.names = FALSE)

# $1,000,000 marketing plan.
budget <- customers %>% group_by(segment) %>% summarise(customers = n(), predicted_return_rate = mean(predicted_return_probability), value_measure = median(monetary), .groups = "drop") %>% mutate(marketing_action = case_when(segment == "High-Value Loyal Customers" ~ "Recognition, early access, personalized cross-sell", segment == "Middle-Value Growth Customers" ~ "A/B-test bundles, cross-sell, and moderate incentives", segment == "Recently Lapsed / At-Risk Customers" ~ "Low-cost win-back test with response measurement", TRUE ~ "Automated reminders and low-cost seasonal communication"), allocation = case_when(segment == "High-Value Loyal Customers" ~ 250000, segment == "Middle-Value Growth Customers" ~ 400000, segment == "Recently Lapsed / At-Risk Customers" ~ 250000, TRUE ~ 100000), percent_budget = allocation / 1000000, justification = "Balances value protection, churn risk, growth potential, and contact cost.")
write.csv(budget, "outputs/budget_allocation.csv", row.names = FALSE)
write.csv(customers, "outputs/customer_level_features.csv", row.names = FALSE)

# Required visualizations.
ggsave("figures/pca_variance.png", ggplot(pca_variance, aes(component, cumulative_variance, group = 1)) + geom_line() + geom_point() + geom_hline(yintercept = .8, linetype = 2) + labs(title = "PCA cumulative explained variance", x = "Principal component", y = "Cumulative variance"), width = 7, height = 4)
ggsave("figures/k_diagnostics.png", ggplot(diagnostics, aes(k, silhouette)) + geom_line() + geom_point() + labs(title = "K-means silhouette diagnostic", x = "Number of clusters", y = "Average silhouette width"), width = 7, height = 4)
ggsave("figures/cluster_map.png", ggplot(customers, aes(PC1, PC2, color = segment)) + geom_point(alpha = .6) + labs(title = "Customer segments in PCA space"), width = 8, height = 5)
