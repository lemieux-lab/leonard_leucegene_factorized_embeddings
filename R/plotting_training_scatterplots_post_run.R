library(dplyr)
library(ggplot2)
library(tidyr)
library(ggpubr)

args = commandArgs(trailingOnly = TRUE)
wd = args[1]
mid = args[2]

filename = paste(wd, mid, "y_true_pred_all.txt", sep = "/")
data = read.csv(filename)
fit = lm(data$y_pred ~ data$y_true)
plot1 = ggplot(data, aes(x = y_true, y = y_pred)) + 
  geom_hex(bins = 200)+ scale_fill_gradient(low = "white", high = "blue") +  
  #geom_smooth(method='lm', se = FALSE, color = "grey", linetype = "dashed") +
  labs(title = paste("Adj R2 = ", signif(summary(fit)$adj.r.squared, 5),
                     "Intercept =",signif(fit$coef[[1]],5 ),
                     " Slope =",signif(fit$coef[[2]], 5),
                     " P =",signif(summary(fit)$coef[2,4], 5))) + 
  scale_x_continuous(limits = c(-1,1))+
  scale_y_continuous(limits = c(-1,1))
  

svg(paste(wd, paste(mid, "training_scatterplot_postrun.svg", sep ="_"), sep = "/"))
plot1
dev.off()
