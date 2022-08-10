library(dplyr)
library(ggplot2)
library(tidyr)


basepath = "/u/sauves/leonard_leucegene_factorized_embeddings/RES/EMBEDDINGS"
dirs = list.dirs(basepath, recursive = F)

args = commandArgs(trailingOnly = TRUE)
wd = args[1]
mid = args[2]

#CDS_tsne = read.csv(paste(wd, paste(mid, "_CDS_tsne_df.txt", sep = ""), sep = "/"))
FE_tsne = read.csv(paste(wd, paste(mid, "_train_test_FE_tsne.txt", sep = ""), sep = "/") ) %>% mutate(method = "FE")
#LSC17_tsne = read.csv(paste(wd, paste(mid, "_lsc17_tsne_df.txt", sep = ""), sep = "/"))
PCA_tsne = read.csv(paste(wd, paste(mid, "_train_test_PCA_tsne.txt", sep = ""), sep = "/")) %>% mutate(method="PCA")
merged = rbind(FE_tsne, PCA_tsne) 
#merged = CDS_tsne %>% rbind(LSC17_tsne) %>% rbind(PCA_tsne) %>% rbind(FE_tsne)   
p = ggplot(merged, aes(x = tsne_1, y = tsne_2, col = interest_group, shape = train_test)) + geom_point() + coord_fixed() + 
  theme_classic() + scale_color_manual(values = c("orange",  "darkcyan", "grey", "magenta"))  + facet_grid(.~method)
g = ggplot(merged, aes(x = tsne_1, y = tsne_2, col = cyto_group, shape = train_test)) + geom_point() + coord_fixed() + 
  theme_classic() + facet_grid(.~method)

svg(paste(wd, paste(mid,"_test_tsne_t821_inv16.svg", sep=""), sep ="/"), width = 20, height = 5)
p
dev.off()

svg(paste(wd, paste(mid, "_test_tsne_cyto_group.svg",sep=""), sep ="/"), width = 20, height = 5)
g
dev.off()


tst_losses = read.csv(paste(wd, mid, "tst_loss.txt", sep = "/"))
h = ggplot(tst_losses, aes(x = epoch, y = loss)) + geom_line()
svg(paste(wd, paste(mid, "_tst_loss.svg",sep=""), sep ="/"), width=5, height =5)
h
dev.off()