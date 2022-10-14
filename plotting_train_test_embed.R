library(dplyr)
library(ggplot2)
library(tidyr)

setwd("/u/sauves/leonard_leucegene_factorized_embeddings/RES/EMBEDDINGS")
filename = "embeddings_2022-10-14T13:27:16.946/FE_01f209c57d002ecf636b0_fold1_train_test.csv"
data = read.csv(filename)
names(data)
g = ggplot(data, aes(x = embed1, y = embed2, col = Cytogenetic.group, shape = factor(train))) + 
  geom_point(size = 5, alpha = 0.5) +
  coord_fixed() + 
  # scale_color_manual(values = c("orange", "darkcyan", "grey", "magenta")) +
  theme_classic()

svg("embeddings_2022-10-14T13:27:16.946/FE_01f209c57d002ecf636b0_fold1_train_test.svg", width = 15, height = 7)
g
dev.off()
