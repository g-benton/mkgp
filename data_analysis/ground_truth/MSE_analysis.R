library(ggplot2)
#library(cowplot)
library(dplyr)
library(tidyr)
setwd("~/Google Drive/Fall 18/ORIE6741/mkgp/data_analysis/compare-kernels/holdout-data/")
#theme_set(theme_bw())

tasks <- "example1.csv"
mse_unif <- "all_mse.csv"
mse_int <- "all_interval_mse.csv"

mse_plots <- function(tasks,mse_unif,mse_int) {
  ex1_tasks <- read.csv(tasks,header=F) %>%
    gather(task,y,2:3)
  
  task_plot <- ggplot(ex1_tasks,aes(x=V1,y=y,color=task)) +
    scale_color_discrete(guide=F) +
    scale_y_continuous(labels=NULL,breaks=NULL) +
    labs(x=NULL,y=NULL) +
    theme_bw() +
    geom_line()
  
  ex1_unif <- read.csv(mse_unif,header=F)
  names(ex1_unif) <- c("meth","task","mse")
  ex1_unif$train <- "uniform"
  ex1_int <- read.csv(mse_int,header=F)
  names(ex1_int) <- c("meth","task","mse")
  ex1_int$train <- "interval"
  ex1 <- rbind(ex1_unif,ex1_int) %>%
    group_by(meth,task,train) %>%
    summarize(mean=mean(mse),
              se=sd(mse)/n())
  ex1$meth <- relevel(ex1$meth,ref="Multi")
  ex1$task <- as.factor(ex1$task)
  
  mse_plot <- ggplot(ex1,aes(x=mean,color=task)) +
    geom_point(aes(x=mean,y=0)) +
    geom_errorbarh(aes(xmin=mean-2*se,xmax=mean+2*se,y=0)) +
    labs(x="MSE") +
    scale_y_continuous(name=NULL,labels=NULL,breaks = NULL) +
    scale_color_discrete(guide=F) +
    theme_bw() +
    facet_grid(meth~train)
  list(cowplot::plot_grid(task_plot,mse_plot,nrow=2,rel_heights=c(1,2),align="b"))
}

mse_plots("example1.csv","ex1_mse.csv","ex1_interval_mse.csv")
mse_plots("example2.csv","ex2_mse.csv","ex2_interval_mse.csv")
mse_plots("example3.csv","ex3_mse.csv","ex3_interval_mse.csv")
mse_plots("example4.csv","ex4_mse.csv","ex4_interval_mse.csv")
