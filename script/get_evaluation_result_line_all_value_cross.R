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

prediction_dir_Odeg = '../output/prediction/OdegVul/cross-release/'

get.line.level.metrics = function(df.file)
{
  all.gt = df.file$line.level.ground.truth
  all.prob = df.file$prediction.prob
  all.pred = df.file$prediction.label

  # confusion.mat = confusionMatrix(all.pred, reference = all.gt)
  confusion.mat = confusionMatrix(factor(all.pred), reference = factor(all.gt))

  bal.acc = confusion.mat$byClass["Balanced Accuracy"]
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

  eval.result = c(AUC, MCC, bal.acc, F1)

  return(eval.result)
}

get.line.level.eval.result = function(prediction.dir, method.name)
{
  all_files = list.files(prediction.dir)

  all.auc = c()
  all.mcc = c()
  all.bal.acc = c()
  all.f1 = c()

  for(f in all_files) # for looping through files
  {
    df = read.csv(paste0(prediction.dir, f))

    line.level.result = get.line.level.metrics(df)

    AUC = line.level.result[1]
    MCC = line.level.result[2]
    bal.acc = line.level.result[3]
    F1 = line.level.result[4]

    all.auc = append(all.auc,AUC)
    all.mcc = append(all.mcc,MCC)
    all.bal.acc = append(all.bal.acc,bal.acc)
    all.f1 = append(all.f1,F1)
  }

  result.df = data.frame(all.auc,all.mcc,all.bal.acc,all.recall,all.f1)


  all.test.rels = str_replace(all.test.rels, ".csv", "")

  result.df$release = all.test.rels
  result.df$technique = method.name

  return(result.df)
}

Odeg.dp.result = get.line.level.eval.result(prediction_dir_Odeg, "OdegVul")


all.result = rbind(Odeg.dp.result)

names(all.result) = c("AUC","MCC","Balance.Accuracy","F1","Release", "Technique")

auc.result = select(all.result, c("Technique","AUC"))
auc.result = preprocess(auc.result,FALSE)

mcc.result = select(all.result, c("Technique","MCC"))
mcc.result = preprocess(mcc.result,FALSE)

bal.acc.result = select(all.result, c("Technique","Balance.Accuracy"))
bal.acc.result = preprocess(bal.acc.result,FALSE)

f1.result = select(all.result, c("Technique","F1"))
f1.result = preprocess(f1.result,FALSE)

## get within-project result
OdegVul.dp.result$project = c("activemq", "activemq", "activemq", "camel", "camel", "derby", "groovy", "hbase", "hive","jruby", "jruby", "lucene", "lucene", "wicket")

line.level.by.project = OdegVul.dp.result %>% group_by(project) %>% summarise(mean.AUC = mean(all.auc), mean.MCC = mean(all.mcc), mean.bal.acc = mean(all.bal.acc))

names(line.level.by.project) = c("project", "AUC","MCC","Balance.Accuracy","Recall","F1")


## get cross-project result

prediction.dir = '../output/prediction/DeepLineDP/cross-release/'

projs = c('activemq', 'camel', 'derby', 'groovy', 'hbase', 'hive', 'jruby', 'lucene', 'wicket')

all.line.result = NULL


for(p in projs)
{
  actual.pred.dir = paste0(prediction.dir,p,'/')

  all.files = list.files(actual.pred.dir)

  all.auc = c()
  all.mcc = c()
  all.bal.acc = c()
  all.f1 = c()
  all.src.projs = c()
  all.tar.projs = c()

  for(f in all.files)
  {
    df = read.csv(paste0(actual.pred.dir,f))

    f = str_replace(f,'.csv','')
    f.split = unlist(strsplit(f,'-'))
    target = tail(f.split,2)[1]

    df = as_tibble(df)

    df.file = select(df, c(train, test, filename, line.level.ground.truth, prediction.prob, prediction.label))

    df.file = distinct(df.file)

    line.level.result = get.file.level.metrics(df.file)

    AUC = line.level.result[1]
    MCC = line.level.result[2]
    bal.acc = line.level.result[3]
    F1 = line.level.result[4]
    all.auc = append(all.auc, AUC)
    all.mcc = append(all.mcc, MCC)
    all.bal.acc = append(all.bal.acc, bal.acc)
    all.f1 = append(all.f1,F1)
    all.src.projs = append(all.src.projs, p)
    all.tar.projs = append(all.tar.projs,target)

    print(paste0('finished ',f))

  }

  line.level.result = data.frame(all.auc,all.mcc,all.bal.acc,all.f1)
  line.level.result$src = p
  line.level.result$target = all.tar.projs

  all.line.result = rbind(all.line.result, line.level.result)

  print(paste0('finished ',p))

}

final.line.level.result = all.line.result %>% group_by(target) %>% summarize(auc = mean(all.auc), balance_acc = mean(all.bal.acc), mcc = mean(all.mcc), f1 = (all.f1))


