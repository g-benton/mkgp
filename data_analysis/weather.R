w <- read.csv("data/weath.csv")
cw <- w %>%
  filter(STATION=="USC00304174")
cw$DATE <- as.Date(cw$DATE)


  
