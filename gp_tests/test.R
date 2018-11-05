library(MASS)
library(dplyr)
library(ggplot2)

d_full <- data.frame(x=seq(0,20,.01)) %>%
  mutate(f=sin(x) + cos(x/2))
d_full$y <- d_full$f + rnorm(nrow(d_full),0,1)

d <- d_full[sample(1:nrow(d_full),30),]
#d <- d_full

d %>%
  ggplot(aes(x=x,y=y)) +
  geom_point() +
  geom_line(data=d_full,aes(x=x,y=f),color="blue")

rbf <- function(x1,x2,l,a) {
  a^2*exp(-(x1-x2)^2/l^2)
}

l <- 2

K_X_X <- matrix(nrow=nrow(d),ncol=nrow(d))
for(i in 1:length(d$x)) {
  for(j in 1:length(d$x)) {
    if(i <= j)
      K_X_X[i,j] <- K_X_X[j,i] <- rbf(d$x[i],d$x[j],l,1)
  }
}

K_Xnew_X <- matrix(nrow=nrow(d_full),ncol=nrow(d))
for(i in 1:nrow(d_full)) {
  for(j in 1:nrow(d)) {
    K_Xnew_X[i,j] <- rbf(d_full$x[i],d$x[j],l,1)
  }
}

K_Xnew_Xnew <- matrix(nrow=nrow(d_full),ncol=nrow(d_full))
for(i in 1:nrow(d_full)) {
  for(j in 1:nrow(d_full)) {
    if(i <= j)
      K_Xnew_Xnew[i,j] <- K_Xnew_Xnew[j,i] <- rbf(d_full$x[i],d_full$x[j],l,1)
  }
}


#d %>%
#  ggplot(aes(x=x,y=y)) +
#  geom_point() +
#  geom_line(aes(y=f),color="blue") +
#  geom_line(data=data.frame(x=d$x,pr=mvrnorm(1,rep(0,length(d$x)),K_X_X)),
#            aes(x=x,y=pr),color="red")+
#  geom_line(data=data.frame(x=d$x,pr=mvrnorm(1,rep(0,length(d$x)),K_X_X)),
#            aes(x=x,y=pr),color="red")+
#  geom_line(data=data.frame(x=d$x,pr=mvrnorm(1,rep(0,length(d$x)),K_X_X)),
#            aes(x=x,y=pr),color="red")

pred_mean <- K_Xnew_X %*% (solve(K_X_X + diag(nrow(K_X_X)))) %*% d$y
pred_cov <- K_Xnew_Xnew -
  K_Xnew_X %*% solve(K_X_X + diag(nrow(d))) %*% t(K_Xnew_X)
f_var <- diag(pred_cov)

d_full %>%
  mutate(f_var=f_var) %>%
  mutate(f_new=pred_mean) %>%
  ggplot(aes(x=x,y=y)) +
  #geom_point(alpha=0.3) +
  geom_ribbon(aes(ymin=f_new-1.96*sqrt(f_var),
                  ymax=f_new+1.96*sqrt(f_var)),
              fill="lightblue",
              alpha=0.3) +
  geom_point(data=d,aes(x=x,y=y),color="red") +
  geom_line(aes(y=f),color="blue") +
  geom_line(aes(y=f_new),color="orange")
