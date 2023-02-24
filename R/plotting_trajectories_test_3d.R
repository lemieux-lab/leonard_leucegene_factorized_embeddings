library(dplyr)
library(ggplot2)
library(tidyr)
library(ggpubr)
library(gg3D)


basepath = "/u/sauves/leonard_leucegene_factorized_embeddings/RES/EMBEDDINGS"

args = commandArgs(trailingOnly = TRUE)
wd = args[1]
mid = args[2]
params_file = read.csv(paste(wd, "model_params.txt", sep = "/"))
nepochs = (params_file %>% filter(modelid == mid))$nepochs

tr_losses = read.csv(paste(wd, mid, "tst_loss.txt", sep = "/"))
tr_l = ggplot(tr_losses, aes(x = epoch, y = loss)) + geom_line()
svg(paste(wd, paste(mid, "_test_loss.svg", sep = "" ), sep = "/"))
tr_l 
dev.off()


for (i in 1:nepochs){
  if (i %% 100 == 0){
    embed = read.csv(paste(wd, mid, paste("test_model_emb_layer_1_epoch_", i, ".txt",sep =""), sep = "/"))
    scatter3d = ggplot(embed, aes(x = emb1, y = emb2, z = emb3, color = group1)) + 
    theme_void() +
    axes_3D() +
    stat_3D() +
    axis_labs_3D() +  
    coord_fixed() + 
    labs_3D(labs = c("emb1", "emb2", "emb3"), hjust=c(0,1,1), vjust=c(1, 1, -0.2), angle=c(0, 0, 90)) +  
    scale_color_manual(values = c("orange", "darkcyan", "grey", "magenta")) +
    ggtitle(paste("epoch ", i, "test loss: ", tr_losses$loss[i]))
    
    outpath = paste(wd,mid, paste("frame", sprintf("%05d", i), "_3d_tst.png", sep = ""), sep = "/")
    loss = ggplot(tr_losses, aes(x = epoch, y = loss)) + geom_line() + theme_classic() + annotate("text", x = i , y= tr_losses$loss[i], colour = "red", label = "o")
    ggarrange(scatter3d, loss, labels = c("A", "B") , ncol = 2, nrow = 1) + bgcolor("white")
    ggsave(outpath, width = 16, height = 8)

  }
}
