library(tidyverse)
library(gridExtra)

library(ModelMetrics)

library(caret)

library(reshape2)
library(pROC)

library(effsize)
library(ScottKnottESD)

# save.fig.dir = '../output/figure/'
save.fig.dir = 'F:/desktop/deep/OdegVul/output/figure/'



dir.create(file.path(save.fig.dir), showWarnings = FALSE)

preprocess <- function(x, reverse){ #  <- 相当于 =
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

get.top.k.tokens = function(df, k)
{
  top.k <- df %>% filter( is.comment.line=="False"  & file.level.ground.truth=="True" & prediction.label=="True" ) %>%
    group_by(test, filename) %>% top_n(k, token.attention.score) %>% select("project","train","test","filename","token") %>% distinct()

  top.k$flag = 'topk'

  return(top.k)
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


prediction_dir_Odeg = 'F:/desktop/deep/OdegVul/output/prediction/DeepLineDP/within-release-Odeg-four/'
prediction_dir_Odeg_Ablation = 'F:/desktop/deep/OdegVul/output/prediction/DeepLineDP/within-release-GCN-four/'


Odeg.dp.result = get.line.level.eval.result(prediction_dir_Odeg, "OdegVul_three")
Odeg_Ablation.dp.result = get.line.level.eval.result(prediction_dir_Odeg_Ablation, "GCN_three")

all.result = rbind(Odeg.dp.result, Odeg_Ablation.dp.result)
print(is.data.frame(all.result))
# names(all.result) = c("AUC","MCC","Balance.Accuracy","Recall","F1","Release", "Technique")
final.line.level.result = all.result %>% group_by(technique) %>% summarize(auc = mean(all.auc,na.rm=TRUE), balance_acc = mean(all.bal.acc,na.rm=TRUE), mcc = mean(all.mcc,na.rm=TRUE), f1 = mean(all.f1,na.rm=TRUE))


