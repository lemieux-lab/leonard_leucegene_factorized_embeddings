library(dplyr)
library(ggplot2)
library(tidyr)
library(ggpubr)


basepath = "/u/sauves/leonard_leucegene_factorized_embeddings/RES/EMBEDDINGS"
dirs = list.dirs(basepath, recursive = F)

args = commandArgs(trailingOnly = TRUE)
wd = args[1]
mid = args[2]
step_size = as.integer(args[3])
params_file =read.csv(paste(wd, "model_params.txt", sep = "/"))
nepochs = (params_file %>% filter(modelid == mid))$nepochs

tr_losses = read.csv(paste(wd, mid, "tr_loss.txt", sep = "/"))
tr_l = ggplot(tr_losses, aes(x = epoch, y = loss)) + geom_line()
svg(paste(wd, paste(mid, "_tr_loss.svg", sep = "" ), sep = "/"))
tr_l 
dev.off()

tr_losses_tiny = tr_losses[lapply(seq(0, max(tr_losses$epoch), step_size), max, 1) %>% unlist,]

cursor = 1

for (i in 1:nepochs){
  if ((i %% step_size == 0) | (i == 1)){
    embed = read.csv(paste(wd, mid, paste("training_model_emb_layer_1_epoch_", i, ".txt",sep =""), sep = "/"))
    
    scatter = ggplot(embed, aes(x = emb1, y = emb2, col = interest_groups)) + geom_point() + 
    theme_classic() + coord_cartesian(xlim = c(-5,5), ylim = c(-5,5)) +
    scale_color_manual(values = c("orange", "darkcyan", "grey", "magenta")) + 
    ggtitle(paste("epoch ", i, "training loss: ", tr_losses$loss[i]))
    outpath = paste(wd,mid, paste("frame", sprintf("%09d", i), "_2d_trn.png", sep = ""), sep = "/")
    loss = ggplot(tr_losses_tiny, aes(x = epoch, y = loss)) + geom_line() + theme_classic() + annotate("text", x = i, y= tr_losses_tiny$loss[cursor], colour = "red", label = "o")
    ggarrange(scatter, loss, labels = c("A", "B") , ncol = 2, nrow = 1)
    ggsave(outpath, width = 16, height = 8)
    cursor = cursor + 1
    
  }
}
