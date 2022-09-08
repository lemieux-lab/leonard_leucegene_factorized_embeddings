library(dplyr)
library(tidyr)
library(ggplot2)

args = commandArgs(trailingOnly = TRUE)
wd = args[1]
mid = args[2]
sampleid = args[3]

fname = paste(sampleid, mid, "pred_expr_corrs", sep = "_")
data = read.csv(paste(wd, paste(fname, ".txt", sep = ""), sep = "/")) %>% select(c("x"="col1", "y"="col2", "corr_pred_expr"="col3", "corr_true_expr"="col4"))
data = data %>% gather(metric_type, correlation, -x, -y)

proj = read.csv(paste(wd, mid, "training_model_emb_layer_1_epoch_12000.txt", sep = "/"))
sample = proj %>% filter(index == sampleid)
MLL_sample_x = sample$emb1
MLL_sample_y = sample$emb2

g = ggplot(data, aes(x=x, y = y)) + geom_point(aes(col = correlation), shape =15) +  
  annotate("text", x = MLL_sample_x, y = MLL_sample_y, label = "x") +
  scale_color_gradient(low = "blue", high = "yellow") + 
  facet_grid(.~metric_type) + coord_fixed() + theme_classic()
svg(paste(wd, paste(fname, ".svg", sep = ""), sep="/"), height = 5, width = 10)
g
dev.off()

