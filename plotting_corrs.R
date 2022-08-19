library(dplyr)
library(tidyr)
library(ggplot2)

basepath = "/u/sauves/leonard_leucegene_factorized_embeddings/RES/EMBEDDINGS"
curr_dir = list.dirs(basepath, recursive = F)[-1]
fname = "04H080_MLL_t_pred_expr_corrs.txt"
data = read.csv(paste(curr_dir, fname, sep = "/"))
data = data %>% gather(metric_type, val, -col1, -col2)
ggplot(data, aes(x=col1, y = col2)) + geom_point(col = val) + facet_grid(~.metric_type)