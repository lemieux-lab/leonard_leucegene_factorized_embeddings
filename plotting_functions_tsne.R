library(dplyr)
library(ggplot2)
library(tidyr)

basepath = "/u/sauves/leonard_leucegene_factorized_embeddings/RES/EMBEDDINGS"
dirs = list.dirs(basepath, recursive = F)
wd = dirs[length(dirs)]

CDS_tsne = read.csv(paste(wd, "CDS_tsne_df.txt", sep = "/"))
FE_tsne = read.csv(paste(wd, "FE_tsne_df.txt", sep = "/"))
LSC17_tsne = read.csv(paste(wd, "lsc17_tsne_df.txt", sep = "/"))
ggplot(CDS_tsne, aes(x = tsne_1, y = tsne_2, col = group)) + geom_point() + coord_fixed() + theme_classic()
ggsave(paste(wd, "out.png", sep ="/"))
