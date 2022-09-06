library(plotly)
library(dplyr)
library(tidyr)

basepath = "/u/sauves/leonard_leucegene_factorized_embeddings/RES/EMBEDDINGS"


args = commandArgs(trailingOnly = TRUE)
wd = args[1]
mid = args[2]

params_file = read.csv(paste(wd, "model_params.txt", sep = "/"))
nepochs = (params_file %>% filter(modelid == mid))$nepochs

embed = read.csv(paste(wd, mid, paste("training_model_emb_layer_1_epoch_", nepochs, ".txt",sep =""), sep = "/"))
fig = plot_ly(embed, x = ~emb1, y = ~emb2, z=~emb3, color = ~interest_groups, colors = c("orange", "darkcyan", "grey", "magenta"))
fig = fig %>% add_markers()
fig

htmlwidgets::saveWidget(as_widget(fig), paste(wd, paste(mid, "_plotly_3d_scatter_by_group_epoch_", nepochs,".html" , sep = ""), sep="/"))
