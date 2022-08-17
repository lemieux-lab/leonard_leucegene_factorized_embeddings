library(dplyr)
library(ggplot2)
library(tidyr)
library(ggpubr)


basepath = "/u/sauves/leonard_leucegene_factorized_embeddings/RES/EMBEDDINGS"
dirs = list.dirs(basepath, recursive = F)

args = commandArgs(trailingOnly = TRUE)
wd = args[1]
mid = args[2]

params_file =read.csv(paste(wd, "model_params.txt", sep = "/"))
nepochs = (params_file %>% filter(modelid == mid))$nepochs

tr_losses = read.csv(paste(wd, mid, "tr_loss.txt", sep = "/"))
tr_l = ggplot(tr_losses, aes(x = epoch, y = loss)) + geom_line()
svg(paste(wd, paste(mid, "_tr_loss.svg", sep = "" ), sep = "/"))
tr_l 
dev.off()

for (i in 1:nepochs){
  if (i %% 100 == 0){
    embed = read.csv(paste(wd, mid, paste("training_model_emb_layer_1_epoch_", i, ".txt",sep =""), sep = "/"))
    
    scatter = ggplot(embed, aes(x = emb1, y = emb2, col = group1)) + geom_point() + 
    theme_classic() + coord_cartesian(xlim = c(-10,10), ylim = c(-10,10)) +
    scale_color_manual(values = c("orange", "darkcyan", "grey", "magenta")) + 
    ggtitle(paste("epoch ", i, "training loss: ", tr_losses$loss[i]))
    outpath = paste(wd,mid, paste("frame", sprintf("%05d", i), "_trn.png", sep = ""), sep = "/")
    loss = ggplot(tr_losses, aes(x = epoch, y = loss)) + geom_line() + theme_classic() + annotate("text", x = i , y= tr_losses$loss[i], colour = "red", label = "o")
    ggarrange(scatter, loss, labels = c("A", "B") , ncol = 2, nrow = 1)
    ggsave(outpath, width = 16, height = 8)

    
  }
}
