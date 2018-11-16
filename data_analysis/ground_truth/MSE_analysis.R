library(ggplot2)
library(dplyr)

ex1 <- read.csv("ex1_mse.csv",header=F)
names(ex1) <- c("meth","task","mse")
ex1_sum <- ex1 %>%
  group_by(meth,task) %>%
  summarize(mean=mean(mse),
         se=sd(mse)/n())
ex1 %>%
  ggplot(aes(x=mse)) +
  geom_histogram(aes(color=meth,fill=meth)) +
  geom_point(data=ex1_sum,
             aes(x=mean,y=0)) +
  geom_vline(data=ex1_sum,
             aes(xintercept=mean)) +
  geom_errorbarh(data=ex1_sum,
             aes(x=NULL,xmin=mean-2*se,xmax=mean+2*se,y=0)) +
  facet_grid(meth~task)

ex2 <- read.csv("ex2_mse.csv",header=F)
names(ex2) <- c("meth","task","mse")
ex2_sum <- ex2 %>%
  group_by(meth,task) %>%
  summarize(mean=mean(mse),
         se=sd(mse)/n())
ex2 %>%
  ggplot(aes(x=mse)) +
  geom_histogram(aes(color=meth,fill=meth)) +
  geom_point(data=ex2_sum,
             aes(x=mean,y=0)) +
  geom_vline(data=ex2_sum,
             aes(xintercept=mean)) +
  geom_errorbarh(data=ex2_sum,
             aes(x=NULL,xmin=mean-2*se,xmax=mean+2*se,y=0)) +
  facet_grid(meth~task)

ex3 <- read.csv("ex3_mse.csv",header=F)
names(ex3) <- c("meth","task","mse")
ex3_sum <- ex3 %>%
  group_by(meth,task) %>%
  summarize(mean=mean(mse),
         se=sd(mse)/n())
ex3 %>%
  ggplot(aes(x=mse)) +
  geom_histogram(aes(color=meth,fill=meth)) +
  geom_point(data=ex3_sum,
             aes(x=mean,y=0)) +
  geom_vline(data=ex3_sum,
             aes(xintercept=mean)) +
  geom_errorbarh(data=ex3_sum,
             aes(x=NULL,xmin=mean-2*se,xmax=mean+2*se,y=0)) +
  facet_grid(meth~task)
