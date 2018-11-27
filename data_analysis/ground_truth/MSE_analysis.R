library(ggplot2)
#library(cowplot)
library(dplyr)
library(tidyr)

tasks <- "example1.csv"
mse_unif <- "ex1_mse.csv"
mse_int <- "ex1_interval_mse.csv"

mse_plots <- function(tasks,mse_unif,mse_int) {
  ex1_tasks <- read.csv(tasks,header=F,stringsAsFactors = F) %>%
    gather(task,y,2:3)
  
  task_plot <- ggplot(ex1_tasks,aes(x=V1,y=y,color=task)) +
    scale_color_manual(values=c("#1F77B4","#FF7F0E"),guide=F) +
    #scale_y_continuous(labels=NULL,breaks=NULL) +
    labs(x=NULL,y=NULL) +
    theme_bw() +
    geom_line()
  
  ex1_unif <- read.csv(mse_unif,header=F,stringsAsFactors = F)
  names(ex1_unif) <- c("meth","task","mse")
  ex1_unif$train <- "uniform"
  ex1_int <- read.csv(mse_int,header=F,stringsAsFactors = F)
  names(ex1_int) <- c("meth","task","mse")
  ex1_int$train <- "interval"
  ex1 <- rbind(ex1_unif,ex1_int) %>%
    mutate(meth=ifelse(meth!="Multi",meth,"MK")) %>%
    mutate(meth=ifelse(meth=="Simple","Indep",meth)) %>%
    group_by(meth,task,train) %>%
    summarize(mean=mean(mse),
              se=sd(mse)/n())
  ex1_pool <- rbind(ex1_unif,ex1_int) %>%
    mutate(meth=ifelse(meth!="Multi",meth,"MK")) %>%
    mutate(meth=ifelse(meth=="Simple","Indep",meth)) %>%
    group_by(meth,train) %>%
    summarize(mean=mean(mse),
              se=sd(mse)/n())
  ex1$meth <- as.factor(ex1$meth)
  ex1$meth <- relevel(ex1$meth,ref="MK")
  ex1_pool$meth <- as.factor(ex1_pool$meth)
  ex1_pool$meth <- relevel(ex1_pool$meth,ref="MK")
  ex1$task <- as.factor(ex1$task)
  
  
  mse_plot <- ggplot(ex1,aes(x=mean)) +
    geom_point(aes(x=mean,y=0,color=task)) +
    geom_point(data=ex1_pool,aes(x=mean,y=0)) +
    geom_errorbarh(aes(xmin=mean-2*se,xmax=mean+2*se,y=0,color=task)) +
    geom_errorbarh(data=ex1_pool,aes(xmin=mean-2*se,xmax=mean+2*se,y=0)) +
    labs(x="MSE") +
    scale_y_continuous(name=NULL,labels=NULL,breaks = NULL) +
    theme_bw() +
    scale_color_manual(values=c("#1F77B4","#FF7F0E"),guide=F) +
    facet_grid(meth~train,scales="free")
  #list(cowplot::plot_grid(task_plot,mse_plot,nrow=2,rel_heights=c(1,2),align="b"))
  mse_plot
}

mse_plots("example5.csv","ex5_mse.csv","ex5_interval_mse.csv")

mse_plots("example1.csv","ex1_mse.csv","ex1_interval_mse.csv")
mse_plots("example2.csv","ex2_mse.csv","ex2_interval_mse.csv")
mse_plots("example3.csv","ex3_mse.csv","ex3_interval_mse.csv")
mse_plots("example4.csv","ex4_mse.csv","ex4_interval_mse.csv")
