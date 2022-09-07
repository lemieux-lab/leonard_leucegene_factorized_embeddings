library(plotly)
library(dplyr)
library(tidyr)

basepath = "/u/sauves/leonard_leucegene_factorized_embeddings/RES/EMBEDDINGS"

wd = "embeddings_2022-09-07T14:24:12.177"
mid = "FE_e7bd3d6c9a0a9f8d7092d"

#params_file = read.csv(paste(basepath, wd, "model_params.txt", sep = "/"))
#nepochs = (params_file %>% filter(modelid == mid))$nepochs
nepochs = "1000000"
embed = read.csv(paste(basepath, wd, mid, paste("training_model_emb_layer_1_epoch_", nepochs, ".txt",sep =""), sep = "/"))
embed = embed %>% mutate(npm1 = ifelse(npm1=="wt", 4, 3))
embed = embed %>% mutate(sex = ifelse(sex=="F", 4, 3))

fig = plot_ly(embed, x = ~emb1, y = ~emb2, z=~emb3, color = ~interest_groups, size= ~npm1, colors = c("orange", "darkcyan", "grey", "magenta"))
fig = fig %>% add_markers()
fig

htmlwidgets::saveWidget(as_widget(fig), paste(basepath, wd, paste(mid, "_plotly_3d_scatter_by_group_size_npm1_epoch_", nepochs,".html" , sep = ""), sep="/"))
