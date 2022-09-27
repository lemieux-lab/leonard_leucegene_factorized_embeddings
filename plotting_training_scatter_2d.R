library(dplyr)
library(ggplot2)
library(tidyr)
library(ggpubr)


basepath = "/u/sauves/leonard_leucegene_factorized_embeddings/RES/EMBEDDINGS"
dirs = list.dirs(basepath, recursive = F)

args = commandArgs(trailingOnly = TRUE)
wd = args[1]
mid = args[2]
# wd = paste(basepath, "embeddings_2022-09-21T22:55:02.452", sep = "/")
# mid = "FE_a5122a4e2f39da2e5ae34"
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

embed = read.csv(paste(wd, mid, paste("training_model_emb_layer_1_epoch_", 1, ".txt",sep =""), sep = "/"))
scatter = ggplot(embed, aes(x = emb1, y = emb2, col = interest_groups)) + geom_point() + 
    theme_classic() + coord_fixed() +
    scale_color_manual(values = c("orange", "darkcyan", "grey", "magenta")) + 
    ggtitle(paste("epoch ", 1, "training loss: ", tr_losses$loss[1]))
svg(paste(wd,mid, paste("frame", sprintf("%09d", 1), "_2d_trn.svg", sep = ""), sep = "/"), width = 10, height = 10)
    scatter
dev.off()
embed = read.csv(paste(wd, mid, paste("training_model_emb_layer_1_epoch_", nepochs, ".txt",sep =""), sep = "/"))
scatter = ggplot(embed, aes(x = emb1, y = emb2, col = interest_groups)) + geom_point() + 
    theme_classic() + coord_fixed() +
    scale_color_manual(values = c("orange", "darkcyan", "grey", "magenta")) + 
    ggtitle(paste("epoch ", nepochs, "training loss: ", tr_losses$loss[nepochs]))
svg(paste(wd,mid, paste("frame", sprintf("%09d", nepochs), "_2d_trn.svg", sep = ""), sep = "/"),  width = 10, height = 10)
    scatter
dev.off()
embed = read.csv(paste(wd, mid, paste("training_model_emb_layer_1_epoch_", nepochs, ".txt",sep =""), sep = "/"))

# print(embed %>% group_by(cyto_group) %>% summarise(n = n()))
# print(embed %>% group_by(interest_groups) %>% summarise(n = n()))
embed$shape_lvl = factor(embed$cyto_group)
scatter = ggplot(embed, aes(x = emb1, y = emb2, col = cyto_group, shape = cyto_group)) + geom_point(size = 10) + 
    theme_classic() + coord_fixed() + scale_shape_manual(values = 13:26) + 
    ggtitle(paste("epoch ", nepochs, "training loss: ", tr_losses$loss[nepochs]))
svg(paste(wd,mid, paste("frame", sprintf("%09d", nepochs), "_by_group_2d_trn.svg", sep = ""), sep = "/"),  width = 20, height = 10)
    scatter
dev.off()
scatter = ggplot(embed, aes(x = emb1, y = emb2, col = cyto_group)) + geom_text(aes(label = index), size = 1) + 
    theme_classic() + coord_fixed() + 
    ggtitle(paste("epoch ", nepochs, "training loss: ", tr_losses$loss[nepochs]))
svg(paste(wd,mid, paste("frame", sprintf("%09d", nepochs), "_by_id_2d_trn.svg", sep = ""), sep = "/"),  width = 20, height = 10)
scatter
dev.off()