library(ggplot2)
library(tidyr)
library(dplyr)
phk <- read.csv("data/PHKHistoricalPricing.csv",header=F,stringsAsFactors = F)
phk$t <- 1:nrow(phk)
  
phk <- phk %>%
  select(V2,V5,t) %>%
  gather(meas,price,V2:V5)

phk %>%
  filter(t<4100) %>%
  filter(t>2800) %>%
  ggplot(aes(x=t,y=price,color=meas)) +
  geom_line() +
  scale_x_continuous(breaks=NULL) +
  labs(y="Price (USD)",x="Time") +
  scale_color_manual(NULL,values=c("#FF7F0E","#1F77B4"),labels=c("NAV","Market Price")) +
  theme_bw()

phk <- phk %>%
  group_by(meas) %>%
  mutate(return = (price - lag(price))/price)

phk %>% filter(t<4100) %>%
  filter(t>3800) %>%
  ggplot(aes(x=t,y=abs(return),color=meas)) +
  facet_grid(meas~.) +
  geom_line()
