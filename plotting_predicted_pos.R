library(dplyr)
library(tidyr)
library(ggplot2)

args = commandArgs(trailingOnly = TRUE)
wd = args[1]
mid = args[2]
sampleid = args[3]
pos_x = as.numeric(args[4])
pos_y = as.numeric(args[5])

fname = paste(sampleid, mid, "inferred_positions", sep = "_")
data = read.csv(paste(wd, paste(fname, ".txt", sep = ""), sep = "/"))
g = ggplot(data, aes(x = x, y = y)) + geom_point() +   
  annotate("text", x = pos_x,y = pos_y, label = "X", color = "red") + 
  theme_classic() +
  scale_x_continuous(limits = c(-4,4)) +
  scale_y_continuous(limits = c(-4,4))
svg(paste(wd, paste(fname, ".svg", sep = ""), sep = "/"), width = 7, height = 7)
g
dev.off()