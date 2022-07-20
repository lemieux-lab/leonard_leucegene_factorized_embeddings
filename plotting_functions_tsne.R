library(dplyr)
library(ggplot2)
library(tidyr)

basepath = "/u/sauves/leonard_leucegene_factorized_embeddings/RES/EMBEDDINGS"
dirs = list.dirs(basepath, recursive = F)
wd = dirs[length(dirs)-2]

CDS_tsne = read.csv(paste(wd, "CDS_tsne_df.txt", sep = "/"))
FE_tsne = read.csv(paste(wd, "FE_tsne_df.txt", sep = "/"))
LSC17_tsne = read.csv(paste(wd, "lsc17_tsne_df.txt", sep = "/"))
PCA_tsne = read.csv(paste(wd, "PCA_tsne_df.txt", sep = "/"))

merged = CDS_tsne %>% rbind(LSC17_tsne) %>% rbind(PCA_tsne) %>% rbind(FE_tsne)   
p = ggplot(merged, aes(x = tsne_1, y = tsne_2, col = interest_group)) + geom_point() + coord_fixed() + 
  theme_classic() + facet_grid(.~method) + scale_color_manual(values = c("red", "grey", "blue"))
g = ggplot(merged, aes(x = tsne_1, y = tsne_2, col = cyto_group)) + geom_point() + coord_fixed() + 
  theme_classic() + facet_grid(.~method) 

svg(paste(wd, "tsne_t821_inv16.svg", sep ="/"), width = 20, height = 5)
p
dev.off()

svg(paste(wd, "tsne_cyto_group.svg", sep ="/"), width = 20, height = 5)
g
dev.off()
