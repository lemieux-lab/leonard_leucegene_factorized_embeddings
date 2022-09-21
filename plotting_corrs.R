library(dplyr)
library(tidyr)
library(ggplot2)

args = commandArgs(trailingOnly = TRUE)
wd = args[1]
mid = args[2]
sampleid = args[3]
epoch_nb = as.character(args[4])


fname = paste(sampleid, mid, "pred_expr_corrs", sep = "_")
data = read.csv(paste(wd, paste(fname, ".txt", sep = ""), sep = "/")) %>% select(c("x"="col1", "y"="col2", "mse_norm"="col3", "corr_true_expr"="col4"))
#data = data %>% gather(metric_type, correlation, -x, -y)

proj = read.csv(paste(wd, mid, paste("training_model_emb_layer_1_epoch_",epoch_nb, ".txt", sep = ""), sep = "/"))
# sample = proj %>% filter(index == sampleid)
# MLL_sample_x = sample$emb1
# MLL_sample_y = sample$emb2
# corr_true = data %>% filter(metric_type == "corr_true_expr") 
# max_corr = which.max(corr_true$correlation)

g = ggplot(data, aes(x=x, y = y)) + geom_point(aes(col = corr_true_expr))+scale_color_gradient(low = "blue", high = "yellow") + coord_fixed() + theme_classic()
# g = ggplot(data, aes(x=x, y = y)) + geom_point(aes(col = correlation), shape =15) +  
  # annotate("text", x = MLL_sample_x, y = MLL_sample_y, label = "x") +
  # annotate("text", x = -2.48, y = -0.36, label = "x", color = "red") +
  # annotate("text", x = corr_true[max_corr,]$x, y = corr_true[max_corr,]$y, label = "x", color = "green")+
  # scale_color_gradient(low = "blue", high = "yellow") +
  # coord_fixed()+
  # theme_classic()
  # facet_grid(.~metric_type) + coord_fixed() + theme_classic()
svg(paste(wd, paste(fname, ".svg", sep = ""), sep="/"), height = 5, width = 10)
g
dev.off()

