library(dplyr)
library(ggplot2)
library(tidyr)


basepath = "/u/sauves/leonard_leucegene_factorized_embeddings/RES/EMBEDDINGS"
dirs = list.dirs(basepath, recursive = F)

args = commandArgs(trailingOnly = TRUE)
wd = args[1]
mid = args[2]

CDS_tsne_2d = read.csv(paste(wd, paste(mid, "_CDS_train_tsne_df.txt", sep = ""), sep = "/"))
FE_2d = read.csv(paste(wd, paste(mid, "_CDS_train_FE_df.txt", sep = ""), sep = "/"))
PCA_2d = read.csv(paste(wd, paste(mid, "_CDS_train_PCA_1_2_df.txt", sep = ""), sep = "/"))

merged = CDS_tsne_2d %>% rbind(PCA_2d) %>% rbind(FE_2d)   
p = ggplot(merged, aes(x = dim_1, y = dim_2, col = interest_group)) + geom_point() + coord_fixed() + 
  theme_classic() + facet_grid(.~method) + scale_color_manual(values = c("orange",  "darkcyan", "grey", "magenta"))
# g = ggplot(merged, aes(x = tsne_1, y = tsne_2, col = cyto_group)) + geom_point() + coord_fixed() + 
#  theme_classic() + facet_grid(.~method) 

svg(paste(wd, paste(mid,"_train_tsne_t821_inv16.svg", sep=""), sep ="/"), width = 20, height = 5)
p
dev.off()
