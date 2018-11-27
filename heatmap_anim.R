library(dplyr)
library(ggplot2)
library(gganimate)

cust_theme <- theme_bw() + theme(
  #legend.position="none", 
  axis.title = element_blank(), axis.ticks = element_blank(), 
  axis.text = element_blank(), strip.text = element_blank(), 
  strip.background = element_blank(), panel.spacing=unit(0,"lines"),
  panel.border = element_rect(size = 0, color = "black"), 
  panel.grid = element_blank())

# Variance/covariance
v1 <- 1
v2 <- 1
v12 <- -1

# Input space
n <- 20
X <- seq(0,10,length.out=n)

# length-scale for task 1 set at 1, task 2 varies from 1 to 5
l <- seq(1,5,length.out=5)

long <- expand.grid(xi=X,xj=X,task1=1:2,task2=1:2,l2=l)

long <- long %>%
  mutate(l1=1)

calc <- long %>%
  mutate(cov=ifelse(task1==task2,
                    ifelse(task1==1,
                           v1*exp(-0.5*((xi-xj)/l1)^2),
                           v2*exp(-0.5*((xi-xj)/l2)^2)),
                    v12*sqrt(2*l1*l2/(l1^2+l2^2))*exp(-(xi-xj)^2/(l1^2+l2^2))))

test <- ggplot(calc,aes(x=xj,y=xi,fill=cov)) +
  geom_tile() +
  scale_y_reverse() +
  scale_x_continuous() +
  scale_fill_gradientn(colours=c("blue","cyan","white", "yellow","red"), values=scales::rescale(c(-1,0,1)))+
  facet_grid(task1~task2) +
  cust_theme +
  transition_states(states = l2,transition_length = 2,state_length = 0) +
  labs(title = 'Task 1 lengthscale: 1; Task 2 lengthscale: {closest_state}', x = 'GDP per capita', y = 'life expectancy')
  
test