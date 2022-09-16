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
nminibatches = as.integer(args[4])
params_file =read.csv(paste(wd, "model_params.txt", sep = "/"))
nepochs = (params_file %>% filter(modelid == mid))$nepochs

tr_losses = read.csv(paste(wd, mid, "tr_loss.txt", sep = "/"))
tr_losses = tr_losses %>% group_by(epochn) %>% summarize(loss = mean(loss))
tr_l = ggplot(tr_losses, aes(x = epochn, y = loss)) + geom_line()
svg(paste(wd, paste(mid, "_tr_loss.svg", sep = "" ), sep = "/"))
tr_l 
dev.off()

#tr_losses_tiny = tr_losses[lapply(seq(0, max(tr_losses$iter), step_size), max, 1) %>% unlist,]



for (i in 1:nepochs){
  if ((i %% step_size == 0) | (i == 1)){
    epoch_number = as.integer(floor((i-1)/nminibatches) + 1)
    embed = read.csv(paste(wd, mid, paste("training_model_emb_layer_1_epoch_", i, ".txt",sep =""), sep = "/"))
    
    scatter = ggplot(embed, aes(x = emb1, y = emb2, col = interest_groups)) + geom_point() + 
    theme_classic() + coord_cartesian(xlim = c(-5,5), ylim = c(-5,5)) +
    scale_color_manual(values = c("orange", "darkcyan", "grey", "magenta")) + 
    ggtitle(paste("epoch ", i, "training loss: ", tr_losses$loss[epoch_number]))
    outpath = paste(wd,mid, paste("frame", sprintf("%09d", i), "_2d_trn.png", sep = ""), sep = "/")
    loss = ggplot(tr_losses, aes(x = epochn, y = loss)) + geom_line() + theme_classic() + annotate("text", x = epoch_number, y= tr_losses$loss[epoch_number], colour = "red", label = "o")
    ggarrange(scatter, loss, labels = c("A", "B") , ncol = 2, nrow = 1)
    ggsave(outpath, width = 16, height = 8)
  
    
  }
}
