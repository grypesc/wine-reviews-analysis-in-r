# Title     : measure_quality
# Objective : Generate ROC, print metrics to console
# Created by: greg
# Created on: 11.11.2020

library(caret)
library(plotROC)


measure_quality <- function(y_predictions, y_true, threshold=0.5) {
  # Generate ROC, print metrics to console
  # threshold - threshold above which qualify to class "1"

  preds_factor <- factor(ifelse(y_predictions >= threshold, 1, 0))
  test_y_factor <- factor(y_true)

  roc.estimate <- calculate_roc(y_predictions, y_true)
  single.rocplot <- ggroc(roc.estimate)
  style_roc(single.rocplot)

  print(confusionMatrix(data = preds_factor, reference = test_y_factor, positive = "1"))
  precision <- posPredValue(preds_factor, test_y_factor, positive="1")
  recall <- sensitivity(preds_factor, test_y_factor, positive="1")
  F1 <- (2 * precision * recall) / (precision + recall)
  print(paste("Precision =", round(precision, 4)))
  print(paste("Recall =", round(recall, 4)))
  print(paste("F1 =", round(F1, 4)))
  print(paste("AUC =", (-1)*round(calc_auc(single.rocplot)$AUC, 4)))
  return (single.rocplot)
}
