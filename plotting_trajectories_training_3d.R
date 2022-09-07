library(dplyr)
library(ggplot2)
library(tidyr)
library(ggpubr)
library(gg3D)


basepath = "/u/sauves/leonard_leucegene_factorized_embeddings/RES/EMBEDDINGS"

args = commandArgs(trailingOnly = TRUE)
wd = args[1]
# wd = "./RES/EMBEDDINGS/embeddings_2022-09-06T10:34:38.932"
mid = args[2]
# mid = "FE_01c2de9c6a83cec0f04cf"
step_size = as.integer(args[3])

params_file = read.csv(paste(wd, "model_params.txt", sep = "/"))
nepochs = (params_file %>% filter(modelid == mid))$nepochs

tr_losses = read.csv(paste(wd, mid, "tr_loss.txt", sep = "/"))
tr_l = ggplot(tr_losses, aes(x = epoch, y = loss)) + geom_line()
svg(paste(wd, paste(mid, "_tr_loss.svg", sep = "" ), sep = "/"))
tr_l 
dev.off()

tr_losses_tiny = tr_losses[lapply(seq(0, max(tr_losses$epoch), step_size), max, 1) %>% unlist,]

cursor = 1


for (i in 1:nepochs){
  if ((i %% step_size == 0) | (i == 1)) {
    embed = read.csv(paste(wd, mid, paste("training_model_emb_layer_1_epoch_", i, ".txt",sep =""), sep = "/"))
    scatter3d = ggplot(embed, aes(x = emb1, y = emb2, z = emb3, color = interest_groups)) + 
    theme_void() +
    axes_3D() +
    stat_3D() +
    axis_labs_3D() +  
    coord_fixed() + 
    labs_3D(labs = c("emb1", "emb2", "emb3"), hjust=c(0,1,1), vjust=c(1, 1, -0.2), angle=c(0, 0, 90)) +  
    scale_color_manual(values = c("orange", "darkcyan", "grey", "magenta")) +
    ggtitle(paste("epoch ", i, "training loss: ", tr_losses$loss[i]))
    

    outpath = paste(wd,mid, paste("frame", sprintf("%09d", i), "_3d_trn.png", sep = ""), sep = "/")
    loss = ggplot(tr_losses_tiny, aes(x = epoch, y = loss)) + geom_line() + theme_classic() + annotate("text", x = i, y= tr_losses_tiny$loss[cursor], colour = "red", label = "o")
    ggarrange(scatter3d, loss, labels = c("A", "B") , ncol = 2, nrow = 1) + bgcolor("white")
    ggsave(outpath, width = 16, height = 8)
    cursor = cursor + 1
  }
  
}
