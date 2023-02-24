library(dplyr)
library(ggplot2)
library(tidyr)

setwd("/u/sauves/leonard_leucegene_factorized_embeddings/RES/EMBEDDINGS")
outdir = "embeddings_2022-10-16T14:25:49.049/FE_bf465440b6909ee943190"
filename = paste(outdir, "_fold1_train_test.csv", sep = "")
data = read.csv(filename) %>% mutate(set = ifelse(train == 1, "train", "test"))
ggplot(data, aes(x = embed1, y = embed2, col = Cytogenetic.group, shape = set)) + 
  geom_point(size = 5, alpha = 0.5) +ggtitle(filename) + 
  scale_x_continuous(breaks = seq(-10,10,2)) + 
  scale_y_continuous(breaks = seq(-10,10,2)) + 
  coord_fixed() +
  theme_classic() + 
  theme(panel.grid.major = element_line(color = "black", size = 0.1, linetype = 2)) +
  theme(panel.grid.minor = element_line(color = "black", size = 0.05, linetype = 2)) 



p = ggplot(data, aes(x = embed1, y = embed2, col = Cytogenetic.group,  shape = set)) + 
  geom_point(size = 1, alpha = 0.5) + 
  geom_text(aes(label = sampleID), size = 1) +
  coord_fixed() +ggtitle(filename) + 
  scale_x_continuous(breaks = seq(-10,10,2)) + 
  scale_y_continuous(breaks = seq(-10,10,2)) + 
  coord_fixed() +
  theme_classic() + 
  theme(panel.grid.major = element_line(color = "black", size = 0.1, linetype = 2)) +
  theme(panel.grid.minor = element_line(color = "black", size = 0.05, linetype = 2)) 



c = ggplot(data, aes(x = embed1, y = embed2, col = interest_groups, shape = set)) + 
  geom_point(size = 4, alpha = 0.5) +
  scale_color_manual(values = c("orange", "darkcyan", "grey", "magenta")) +
  ggtitle(filename) + 
  scale_x_continuous(breaks = seq(-10,10,2)) + 
  scale_y_continuous(breaks = seq(-10,10,2)) + 
  coord_fixed() +
  theme_classic() + 
  theme(panel.grid.major = element_line(color = "black", size = 0.1, linetype = 2)) +
  theme(panel.grid.minor = element_line(color = "black", size = 0.05, linetype = 2)) 
  


svg(paste(outdir, "_fold1_train_test_cyt_group.svg", sep = ""), width = 15, height = 7)
g
dev.off()
svg(paste(outdir, "_fold1_train_test_names.svg", sep = ""), width = 15, height = 7)
p
dev.off()
svg(paste(outdir, "_fold1_train_test_interets_groups.svg", sep = ""), width = 15, height = 7)
c
dev.off()
