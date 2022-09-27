library(dplyr)
library(tidyr)
library(ggplot2)

args = commandArgs(trailingOnly = TRUE)
wd = args[1]
mid = args[2]
sampleid = args[3]
epoch_nb = as.character(args[4])


fname = paste(sampleid, mid, "pred_expr_corrs", sep = "_")
data = read.csv(paste(wd, paste(fname, ".txt", sep = ""), sep = "/")) %>% select(c("x", "y", "mse", "mse_wd", "cor" ))
#data = data %>% gather(metric_type, correlation, -x, -y)

proj = read.csv(paste(wd, mid, paste("training_model_emb_layer_1_epoch_",epoch_nb, ".txt", sep = ""), sep = "/"))
sample = proj %>% filter(index == sampleid)
sample_x = sample$emb1
sample_y = sample$emb2
# corr_true = data %>% filter(metric_type == "corr_true_expr") 
# max_corr = which.max(corr_true$correlation)

g = ggplot(data, aes(x=x, y = y)) + geom_point(aes(col = cor))+scale_color_gradient(low = "blue", high = "yellow") + 
  coord_fixed() + theme_classic() + 
  annotate("text", x = sample_x, y = sample_y, label = "x") 
svg(paste(wd, paste(sampleid, mid, "pred_expr_corrs.svg", sep = ""), sep="/"), height = 5, width = 10)
g
dev.off()

g = ggplot(data, aes(x=x, y = y)) + geom_point(aes(col = log10(mse)))+scale_color_gradient(low = "yellow", high = "blue") + 
  coord_fixed() + theme_classic() + 
  annotate("text", x = sample_x, y = sample_y, label = "x") 
svg(paste(wd, paste(sampleid, mid, "pred_expr_mse.svg", sep = "_"), sep="/"), height =  5, width = 10)
g
dev.off()

g = ggplot(data, aes(x=x, y = y)) + geom_point(aes(col = log10(mse_wd)))+scale_color_gradient(low = "yellow", high = "blue") + 
  coord_fixed() + 
  theme_classic() + 
  annotate("text", x = sample_x, y = sample_y, label = "x") 
svg(paste(wd, paste(sampleid, mid, "pred_expr_mse_wd.svg", sep = "_"), sep="/"), height = 5, width = 10)
g
dev.off()
