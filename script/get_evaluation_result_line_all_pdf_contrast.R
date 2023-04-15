library(tidyverse)
library(gridExtra)

library(ModelMetrics)

library(caret)

library(reshape2)
library(pROC)

library(effsize)
library(ScottKnottESD)

save.fig.dir = '../output/figure_contrast/'

dir.create(file.path(save.fig.dir), showWarnings = FALSE)

preprocess <- function(x, reverse){
  colnames(x) <- c("variable","value")
  tmp <- do.call(cbind, split(x, x$variable))
  tmp <- tmp[, grep("value", names(tmp))]
  names(tmp) <- gsub(".value", "", names(tmp))
  df <- tmp
  ranking <- NULL

  if(reverse == TRUE)
  {
    ranking <- (max(sk_esd(df)$group)-sk_esd(df)$group) +1
  }
  else
  {
    ranking <- sk_esd(df)$group
  }

  x$rank <- paste("Rank",ranking[as.character(x$variable)])
  return(x)
}


get.line.level.metrics = function(df.file)
{
  all.gt = df.file$line.level.ground.truth
  all.prob = df.file$prediction.prob
  all.pred = df.file$prediction.label

  # confusion.mat = confusionMatrix(all.pred, reference = all.gt)
  confusion.mat = confusionMatrix(factor(all.pred), reference = factor(all.gt))

  bal.acc = confusion.mat$byClass["Balanced Accuracy"]
  Recall = confusion.mat$byClass["Recall"]
  F1 = confusion.mat$byClass["F1"]

  AUC = pROC::auc(all.gt, all.prob)

  all.pred[all.pred=="False"] = 0
  all.pred[all.pred=="True"] = 1
  all.gt[all.gt=="False"] = 0
  all.gt[all.gt=="True"] = 1

  # all.gt = as.numeric_version(all.gt)
  all.gt = as.numeric(all.gt)

  # all.pred = as.numeric_version(all.pred)
  all.pred = as.numeric(all.pred)

  MCC = mcc(all.gt, all.pred, cutoff = 0.5)

  if(is.nan(MCC))
  {
    MCC = 0
  }

  eval.result = c(AUC, MCC, bal.acc, Recall, F1)

  return(eval.result)
}

get.line.level.eval.result = function(prediction.dir, method.name)
{
  all_files = list.files(prediction.dir)

  all.auc = c()
  all.mcc = c()
  all.bal.acc = c()
  all.recall = c()
  all.f1 = c()
  all.test.rels = c()

  for(f in all_files) # for looping through files
  {
    df = read.csv(paste0(prediction.dir, f))

    line.level.result = get.line.level.metrics(df)

    AUC = line.level.result[1]
    MCC = line.level.result[2]
    bal.acc = line.level.result[3]
    Recall = line.level.result[4]
    F1 = line.level.result[5]

    all.auc = append(all.auc,AUC)
    all.mcc = append(all.mcc,MCC)
    all.bal.acc = append(all.bal.acc,bal.acc)
    all.recall = append(all.recall,Recall)
    all.f1 = append(all.f1,F1)
    all.test.rels = append(all.test.rels,f)

  }

  result.df = data.frame(all.auc,all.mcc,all.bal.acc,all.recall,all.f1)


  all.test.rels = str_replace(all.test.rels, ".csv", "")

  result.df$release = all.test.rels
  result.df$technique = method.name

  return(result.df)
}


prediction_dir_Odeg = '../output/prediction/DeepLineDP/within-release-Odeg/'
prediction_dir_SAGE = '../output/prediction/DeepLineDP/within-release-SAGE/'
prediction_dir_GAT = '../output/prediction/DeepLineDP/within-release-GAT/'
prediction_dir_GCN = '../output/prediction/DeepLineDP/within-release-GCN/'
prediction_dir_DeepLineDP = '../output/prediction/DeepLineDP/within-release-DeepLineDP/'
prediction_dir_LSTM = '../output/prediction/DeepLineDP/within-release-Lstm/'
prediction_dir_GGNN = '../output/prediction/DeepLineDP/within-release-GGNN/'

Odeg.dp.result = get.line.level.eval.result(prediction_dir_Odeg, "OdegVul")
SAGEVul.dp.result = get.line.level.eval.result(prediction_dir_SAGE, "GraphSage")
GATVul.dp.result = get.line.level.eval.result(prediction_dir_GAT, "LineVD")
GCNVul.dp.result = get.line.level.eval.result(prediction_dir_GCN, "DP-GCNN")
DeepLineDP.dp.result = get.line.level.eval.result(prediction_dir_DeepLineDP, "DeepLine")
LSTM.dp.result = get.line.level.eval.result(prediction_dir_LSTM, "Bi-LSTM")
GGNN.dp.result = get.line.level.eval.result(prediction_dir_GGNN, "GGNN")


all.result = rbind(Odeg.dp.result, GATVul.dp.result, SAGEVul.dp.result, GCNVul.dp.result,
                   DeepLineDP.dp.result,LSTM.dp.result, GGNN.dp.result)

names(all.result) = c("AUC","MCC","Balance.Accuracy","Recall","F1","Release", "Technique")

auc.result = select(all.result, c("Technique","AUC"))
auc.result = preprocess(auc.result,FALSE)

mcc.result = select(all.result, c("Technique","MCC"))
mcc.result = preprocess(mcc.result,FALSE)

bal.acc.result = select(all.result, c("Technique","Balance.Accuracy"))
bal.acc.result = preprocess(bal.acc.result,FALSE)

recall.result = select(all.result, c("Technique","Recall"))
recall.result = preprocess(recall.result,FALSE)

f1.result = select(all.result, c("Technique","F1"))
f1.result = preprocess(f1.result,FALSE)

ggplot(auc.result, aes(x=reorder(variable, -value, FUN=median), y=value)) + geom_boxplot() + facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + ylab("AUC") + xlab("")
ggsave(paste0(save.fig.dir,"line-AUC.png"),width=5.6,height=3.4)

ggplot(bal.acc.result, aes(x=reorder(variable, value, FUN=median), y=value)) + geom_boxplot() + facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + ylab("Balance Accuracy") + xlab("")
ggsave(paste0(save.fig.dir,"line-Balance_Accuracy.png"),width=5.6,height=3.4)

ggplot(mcc.result, aes(x=reorder(variable, value, FUN=median), y=value)) + geom_boxplot()  + facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + ylab("MCC") + xlab("")
ggsave(paste0(save.fig.dir, "line-MCC.png"),width=5.6,height=3.4)

ggplot(f1.result, aes(x=reorder(variable, value, FUN=median), y=value)) + geom_boxplot()  + facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + ylab("F1") + xlab("")
ggsave(paste0(save.fig.dir, "line-F1.pdf"),width=5.6,height=3.4)


