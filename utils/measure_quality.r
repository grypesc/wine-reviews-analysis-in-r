# Title     : measure_quality
# Objective : Generate ROC, print metrics to console
# Created by: greg
# Created on: 11.11.2020

library(caret)
library(pROC)

measure_quality <- function(y_predictions, y_true, threshold=0.5) {
  # Generate ROC with AUC, print metrics to console
  # threshold - threshold above which qualify to class "1"

  preds_factor <- factor(ifelse(y_predictions >= threshold, 1, 0))
  test_y_factor <- factor(y_true)

  plot.roc(test_y, preds, print.auc = TRUE, print.auc.x = 0.15, print.auc.y = 0.05)
  print(confusionMatrix(data = preds_factor, reference = test_y_factor, positive = "1"))
  precision <- posPredValue(preds_factor, test_y_factor, positive="1")
  recall <- sensitivity(preds_factor, test_y_factor, positive="1")
  F1 <- (2 * precision * recall) / (precision + recall)
  print(paste("test precision =", round(precision, 4)))
  print(paste("test recall =", round(recall, 4)))
  print(paste("test  F1 =", round(F1, 4)))

}
