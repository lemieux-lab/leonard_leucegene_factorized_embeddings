library(ggplot2)
library(dplyr)
library(scales)
library(RSvgDevice)
setwd("/u/sauves/leonard_leucegene_factorized_embeddings/")
rdm_accs = read.csv("/u/sauves/leonard_leucegene_factorized_embeddings/RES/EMBEDDINGS/embeddings_2023-02-13T18:30:15.860/TCGA_tst_accs.csv")  %>% mutate(method = "rdm sign")
pca_accs = read.csv("/u/sauves/leonard_leucegene_factorized_embeddings/RES/EMBEDDINGS/embeddings_2023-02-13T18:30:15.860/TCGA_pca_tst_accs.csv") %>% mutate(method = "pca")
data = rbind(rdm_accs, pca_accs)  

title = "Logistic Regression dans Flux sur TCGA : détermination des 30 types de cancer. 
Échantillons: 17382. Performance vs nb input features (Gene Expression RNA-Seq log10(tpm+1)). PCA vs random signatures. 
Pas de régularization. 5-Fold Cross-val (test n=3400 (20%)). crossentropy loss."
g = ggplot(data, aes(x = lengths, y = tst_acc * 100, fill = method)) + geom_point(color = "black", shape = 21) +
  theme_light() + 
  scale_x_continuous(trans = log10_trans(), breaks = unique(data$lengths)) +
  scale_y_continuous(limits = c(0,100))+
  # breaks = trans_breaks("log10", function(x) 10^x),
  # labels = trans_format("log10", math_format(10^.x))) +
  # scale_x_log10(breaks = trans_breaks("log10", function(x) 10^ x), 
  #               labels = trans_format("log10", math_format(10^.x))) + 
  #annotation_logticks() + 
  xlab("Number of genes (randomly picked) - log10 scale") + 
  ylab("Accuracy % on test set") + 
  ggtitle(title) + 
  theme(text = element_text(size = 14, color = "black"),
        title = element_text(size = 10, color = "black") ) 

ggsave("/u/sauves/leonard_leucegene_factorized_embeddings/RES/EMBEDDINGS/embeddings_2023-02-13T18:30:15.860/TCGA_tst_accs_pca_vs_rdm.png")
pdf("/u/sauves/leonard_leucegene_factorized_embeddings/RES/EMBEDDINGS/embeddings_2023-02-13T18:30:15.860/TCGA_tst_accs_pca_vs_rdm.pdf", width = 10, height = 5)
g
dev.off()

