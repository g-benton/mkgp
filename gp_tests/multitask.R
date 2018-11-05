library(dplyr)
library(ggplot2)
library(MASS)
# functions
f1 <- function(x) {
  80*sin(2*pi*x/10)
}
f2 <- function(x) {
  40 + f1(x) + 3*sin(2*pi*x)
}
n <- 80
D <- data.frame(task=rep(c("one","two"),each=n),
           x=rep(c(runif(n/2,0,2.5),runif(n/2,3,5)),2)) %>%
  mutate(y=ifelse(task=="one",f1(x),f2(x)))
D$y <- D$y+rnorm(2*n,0,3)
plot(D$x,D$y,col=D$task)
curve(f1,0,10,1000,col="green",add=T)
curve(f2,0,10,1000,col="purple",add=T)



# first just try to draw from prior
k <- function(x1,x2,l,a) {
  a^2*exp(-0.5*((x1 - x2)/l)^2)
}
kcross <- function(x1,x2,l1,l2,across) {
  across^2*sqrt((2*l1*l2)/(l1^2+l2^2))*
    exp(-(x1-x2)^2/(l1^2+l2^2))
}
#n <- 500
#X <- seq(0,5,length.out=n)
#k1xx <-  k2xx  <- k12xx <- matrix(rep(NA,n),
#                                       nrow=n,
#                                       ncol=n)
l1 <- 2.5
l2 <- 0.3
a1 <- 80
a2 <- 3
across <- 80*3

#for(i in 1:nrow(k1x1x1) ) {
#  for(j in 1:ncol(k1x1x1) ) {
#    if(i <= j) {
#      k1xx[i,j] <- k1xx[j,i] <- k(X[i],X[j],l1,a1)
#      k2xx[i,j] <- k2xx[j,i] <- k(X[i],X[j],l2,a2)
#      k12xx[i,j] <- k12xx[j,i] <- kcross(X[i],X[j],l1,l2,a1,a2)
#    }
#  }
#}

#K <- cbind(rbind(k1xx,
#                 t(k12xx)),
#           rbind(k12xx,
#                 k2xx))
#heatmap(K,Rowv=NA,Colv=NA,scale="none")
#
#y <- mvrnorm(1,rep(0,1000),K_scaled)
#wide <- cbind(y[1:n],y[(n+1):(2*n)])
#matplot(X,wide,type='l')
#legend("topleft",legend=1:2,col=1:2,lty=1:2)

# OK now try to draw posterior mean

# X's are in D$x:
n <- length(D$x)/2
k1xx <- k2xx <- k12xx <- matrix(rep(NA,n^2),
                                nrow=n,
                                ncol=n)
for(i in 1:nrow(k1xx) ) {
  for(j in 1:ncol(k1xx) ) {
    if(i <= j) {
      k1xx[i,j] <- k1xx[j,i] <- k(D$x[i],D$x[j],l1,a1)
      k2xx[i,j] <- k2xx[j,i] <- k(D$x[i],D$x[j],l2,a2)
      k12xx[i,j] <- k12xx[j,i] <- kcross(D$x[i],D$x[j],l1,l2,across)
    }
  }
}
KXX <- cbind(rbind(k1xx,
                 k12xx),
           rbind(k12xx,
                 k2xx))

# We'll evaluate the posterior at the nn=500 points in X
nn <- 500
X <- seq(0,7,length.out=nn)
k1nx <-  k2nx <-  k12nx <- matrix(NA,nrow=n,ncol=nn)

for(i in 1:n ) {
  for(j in 1:nn ) {
      k1nx[i,j] <- k(D$x[i],X[j],l1,a1)
      k2nx[i,j] <- k(D$x[i],X[j],l2,a2)
      k12nx[i,j] <- kcross(D$x[i],X[j],l1,l2,across)
  }
}
KNX <- cbind(rbind(k1nx,
                   k12nx),
             rbind(k12nx,
                   k2nx))


k1nn <- k2nn <- k12nn <- matrix(NA, nrow=nn, ncol=nn)
for(i in 1:nn ) {
  for(j in 1:nn ) {
    if(i <= j) {
      k1nn[i,j] <- k1nn[j,i] <- k(X[i],X[j],l1,a1)
      k2nn[i,j] <- k2nn[j,i] <- k(X[i],X[j],l2,a2)
      k12nn[i,j] <- k12nn[j,i] <- kcross(X[i],X[j],l1,l2,across)
    }
  }
}
KNN <- cbind(rbind(k1nn,
                   k12nn),
             rbind(k12nn,
                   k2nn))


pred_mean <- mean(D$y) + t(KNX) %*% solve(KXX+3*diag(2*n)) %*% (D$y-mean(D$y))
pred_cov <- KNN - t(KNX) %*% solve(KXX + 3*diag(2*n)) %*% KNX
f_var <- diag(pred_cov)

multi_fit <- data.frame(x=rep(X,2),
           task=rep(c("one","two"),each=nn),
           m=pred_mean,
           f_var=f_var)

#ggplot(multi_fit,aes(x=x,y=m,col=task)) +
#  stat_function(fun=f1,n=500,color="black") +
#  stat_function(fun=f2,n=500,color="black") +
#  geom_ribbon(aes(ymin=m-2*sqrt(f_var),
#                  ymax=m+2*sqrt(f_var),
#                  fill=task,linetype=NA),
#              alpha=0.2) +
#  geom_line() +
#  geom_point(data=D,aes(x=x,y=y,color=factor(task)))

# Single-task estimation of the wiggly line
s_pred_mean <- mean(D$y[(n+1):(2*n)]) +
  t(k2nx) %*% solve(k2xx+3*diag(n)) %*% (D$y[(n+1):(2*n)] - mean(D$y[(n+1):(2*n)]))
s_pred_cov <- k2nn - t(k2nx) %*% solve(k2xx + 3*diag(n)) %*% k2nx
s_f_var <- diag(s_pred_cov)
single_fit <- data.frame(x=X,
                         task=rep("two",nn),
                         m=s_pred_mean,
                         f_var=s_f_var)

ggplot(multi_fit,aes(x=x,y=m,col=task)) +
  ylim(NA,150) +
  stat_function(fun=f1,n=500,color="black") +
  stat_function(fun=f2,n=500,color="black") +
  geom_ribbon(aes(ymin=m-2*sqrt(f_var),
                  ymax=m+2*sqrt(f_var),
                  fill=task,linetype=NA),
              alpha=0.2) +
  geom_line() +
  geom_line(data=single_fit,aes(x=x,y=m),color="purple") +
  geom_point(data=D,aes(x=x,y=y,color=factor(task)))
