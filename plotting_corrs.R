library(dplyr)
library(tidyr)
library(ggplot2)

args = commandArgs(trailingOnly = TRUE)
wd = args[1]
mid = args[2]
sampleid = args[3]
epoch_nb = as.character(args[4])


fname = paste(sampleid, mid, "pred_expr_corrs", sep = "_")
data = read.csv(paste(wd, paste(fname, ".txt", sep = ""), sep = "/")) %>% select(c("x"="col1", "y"="col2", "corr_true_expr"="col3", "mse"="col4", "loss"="col5" ))
#data = data %>% gather(metric_type, correlation, -x, -y)

proj = read.csv(paste(wd, mid, paste("training_model_emb_layer_1_epoch_",epoch_nb, ".txt", sep = ""), sep = "/"))
sample = proj %>% filter(index == sampleid)
sample_x = sample$emb1
sample_y = sample$emb2
# corr_true = data %>% filter(metric_type == "corr_true_expr") 
# max_corr = which.max(corr_true$correlation)

g = ggplot(data, aes(x=x, y = y)) + geom_point(aes(col = corr_true_expr))+scale_color_gradient(low = "white", high = "orange") + 
  coord_fixed() + theme_classic() + 
  annotate("text", x = sample_x, y = sample_y, label = "x") 
svg(paste(wd, paste(fname, ".svg", sep = ""), sep="/"), height = 5, width = 10)
g
dev.off()

g = ggplot(data, aes(x=x, y = y)) + geom_point(aes(col = mse))+scale_color_gradient(low = "orange", high = "white") + 
  coord_fixed() + theme_classic() + 
  annotate("text", x = sample_x, y = sample_y, label = "x") 
svg(paste(wd, paste(sampleid, mid, "pred_expr_mse.svg", sep = "_"), sep="/"), height = 5, width = 10)
g
dev.off()

g = ggplot(data, aes(x=x, y = y)) + geom_point(aes(col = loss))+scale_color_gradient(low = "orange", high = "white") + 
  coord_fixed() + 
  theme_classic() + 
  annotate("text", x = sample_x, y = sample_y, label = "x") 
svg(paste(wd, paste(sampleid, mid, "pred_expr_loss.svg", sep = "_"), sep="/"), height = 5, width = 10)
g
dev.off()
