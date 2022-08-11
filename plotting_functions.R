library(dplyr)
library(ggplot2)
library(tidyr)

basepath = "/u/sauves/leonard_leucegene_factorized_embeddings/RES/EMBEDDINGS"
dirs = list.dirs(basepath, recursive = F)

args = commandArgs(trailingOnly = TRUE)
wd = args[1]
mid = args[2]

params_file =read.csv(paste(wd, "model_params.txt", sep = "/"))
nepochs = (params_file %>% filter(modelid == mid))$nepochs

tr_losses = read.csv(paste(wd, mid, "tr_loss.txt", sep = "/"))
ggplot(tr_losses, aes(x = epoch, y = loss)) + geom_line()
for (i in 1:nepochs){
  if (i %% 100 == 0){
    embed = read.csv(paste(wd, mid, paste("model_emb_layer_1_epoch_", i, ".txt",sep =""), sep = "/"))
    
    g = ggplot(embed, aes(x = emb1, y = emb2, col = group1)) + geom_point() + 
    theme_classic() + coord_fixed() + 
    scale_color_manual(values = c("red", "grey", "blue")) + 
    ggtitle(paste("epoch ",i))
    outpath = paste(wd,mid, paste("frame", sprintf("%05d", i), ".png", sep = ""), sep = "/")
    ggsave(outpath)
    
  }
}
