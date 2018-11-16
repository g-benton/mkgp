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

ex4 <- read.csv("ex4_mse.csv",header=F)
names(ex4) <- c("meth","task","mse")
ex4_sum <- ex4 %>%
  group_by(meth,task) %>%
  summarize(mean=mean(mse),
         se=sd(mse)/n())
ex4 %>%
  ggplot(aes(x=mse)) +
  geom_histogram(aes(color=meth,fill=meth)) +
  geom_point(data=ex4_sum,
             aes(x=mean,y=0)) +
  geom_vline(data=ex4_sum,
             aes(xintercept=mean)) +
  geom_errorbarh(data=ex4_sum,
             aes(x=NULL,xmin=mean-2*se,xmax=mean+2*se,y=0)) +
  facet_grid(meth~task)

ex1_interval <- read.csv("ex1_interval_mse.csv",header=F)
names(ex1_interval) <- c("meth","task","mse")
ex1_interval_sum <- ex1_interval %>%
  group_by(meth,task) %>%
  summarize(mean=mean(mse),
         se=sd(mse)/n())
ex1_interval %>%
  ggplot(aes(x=mse)) +
  geom_histogram(aes(color=meth,fill=meth)) +
  geom_point(data=ex1_interval_sum,
             aes(x=mean,y=0)) +
  geom_vline(data=ex1_interval_sum,
             aes(xintercept=mean)) +
  geom_errorbarh(data=ex1_interval_sum,
             aes(x=NULL,xmin=mean-2*se,xmax=mean+2*se,y=0)) +
  facet_grid(meth~task)

ex2_interval <- read.csv("ex2_interval_mse.csv",header=F)
names(ex2_interval) <- c("meth","task","mse")
ex2_interval_sum <- ex2_interval %>%
  group_by(meth,task) %>%
  summarize(mean=mean(mse),
         se=sd(mse)/n())
ex2_interval %>%
  ggplot(aes(x=mse)) +
  geom_histogram(aes(color=meth,fill=meth)) +
  geom_point(data=ex2_interval_sum,
             aes(x=mean,y=0)) +
  geom_vline(data=ex2_interval_sum,
             aes(xintercept=mean)) +
  geom_errorbarh(data=ex2_interval_sum,
             aes(x=NULL,xmin=mean-2*se,xmax=mean+2*se,y=0)) +
  facet_grid(meth~task)

ex3_interval <- read.csv("ex3_interval_mse.csv",header=F)
names(ex3_interval) <- c("meth","task","mse")
ex3_interval_sum <- ex3_interval %>%
  group_by(meth,task) %>%
  summarize(mean=mean(mse),
         se=sd(mse)/n())
ex3_interval %>%
  ggplot(aes(x=mse)) +
  geom_histogram(aes(color=meth,fill=meth)) +
  geom_point(data=ex3_interval_sum,
             aes(x=mean,y=0)) +
  geom_vline(data=ex3_interval_sum,
             aes(xintercept=mean)) +
  geom_errorbarh(data=ex3_interval_sum,
             aes(x=NULL,xmin=mean-2*se,xmax=mean+2*se,y=0)) +
  facet_grid(meth~task)

ex4_interval <- read.csv("ex4_interval_mse.csv",header=F)
names(ex4_interval) <- c("meth","task","mse")
ex4_interval_sum <- ex4_interval %>%
  group_by(meth,task) %>%
  summarize(mean=mean(mse),
         se=sd(mse)/n())
ex4_interval %>%
  ggplot(aes(x=mse)) +
  geom_histogram(aes(color=meth,fill=meth)) +
  geom_point(data=ex4_interval_sum,
             aes(x=mean,y=0)) +
  geom_vline(data=ex4_interval_sum,
             aes(xintercept=mean)) +
  geom_errorbarh(data=ex4_interval_sum,
             aes(x=NULL,xmin=mean-2*se,xmax=mean+2*se,y=0)) +
  facet_grid(meth~task)
