library(dplyr)
require(tidyr)
library(survival)
library(ggplot2)
library(ggfortify)
## non-bio dataset : Rossi 
url = "http://socserv.mcmaster.ca/jfox/Books/Companion/data/Rossi.txt"
Rossi = read.table(url, header=TRUE)
names(Rossi)
mod.allison = coxph(Surv(week, arrest) ~ fin + age + race + wexp + mar + paro + prio, 
                    data=Rossi)
out = summary(mod.allison)
sink("RES/CPH/summary1cph.txt")
print(out)
sink()

Rossi.fin = with(Rossi,
                 data.frame(fin=c(0, 1), 
                            age=rep(mean(age), 2), 
                            race=rep(mean(race == "other"), 2),
                            wexp=rep(mean(wexp == "yes"), 2), 
                            mar=rep(mean(mar == "not married"), 2),
                            paro=rep(mean(paro == "yes"), 2), 
                            prio=rep(mean(prio), 2)))

plot(survfit(mod.allison, newdata = Rossi.fin))
#plot(survfit(mod.allison, newdata = Rossi.fin), conf.int = TRUE, col = c(2,3), lty = 2:3)
legend(30, 0.5, c("financial aid", "no aid"), lty = 2:3)
Rossi.fin
## Leucegene data
setwd("leonard_leucegene_factorized_embeddings/")
Lgn_raw = read.csv("./Data/LEUCEGENE/lgn_pronostic_CF") %>% 
  #filter(Cytogenetic.risk != "favorable cytogenetics") %>% 
  mutate(Cytogenetic.risk.adv = ifelse(Cytogenetic.risk == "adverse cytogenetics", 1, 0))

Lgn_raw %>% group_by(Cytogenetic.risk) %>% summarize(n = n(), npm1 = sum(NPM1.mutation), adv = mean(Cytogenetic.risk.adv))
Lgn = Lgn_raw %>% filter(!(WHO.classification %in% c("Acute megakaryoblastic leukaemia", 
                                                     "Acute erythroid leukaemia", 
                                                     "AML with t(6;9)(p23;q34); DEK-NUP214"))) %>%
  filter(!(Induction_Type %in% c("Experimental palliative therapies")))

mod.Lgn = coxph(Surv(Overall_Survival_Time_days, Overall_Survival_Status) ~ 
                  Cytogenetic.risk + Age_at_diagnosis + FLT3.ITD.mutation + NPM1.mutation + IDH1.R132.mutation + Sex + Tissue, data = Lgn)
out = summary(mod.Lgn)
sink("RES/CPH/summaryLGNcph.txt")
print(out)
sink()

Lgn.NPM1 = with(Lgn, 
                data.frame(NPM1.mutation = c(0,1), 
                           Age_at_diagnosis = rep(mean(Age_at_diagnosis),2),
                           FLT3.ITD.mutation = rep(mean(FLT3.ITD.mutation), 2),
                           IDH1.R132.mutation = rep(mean(IDH1.R132.mutation), 2),
                           Sex = rep(mean(Sex == "F"), 2),
                           Tissue = rep(mean(Tissue == "Blood"),2)
                           )
                )
surv.NPM1 = survfit(mod.Lgn, newdata = Lgn.NPM1)
row.names(surv.NPM1) = c("NPM1.mutation=0", "NPM1.mutation=1")
names(surv.NPM1)
summary(surv.NPM1)
plot(surv.NPM1, conf.int = TRUE, col = c("red", "blue"), lty = 2:3)
legend(3000, .9, c("NPM1 absent", "NPM1 present"), c("red", "blue"), lty = 2:3)

summary(mod.Lgn)
surv.NPM1.2 = survfit(Surv(Overall_Survival_Time_days, Overall_Survival_Status) ~ NPM1.mutation, data = Lgn)
as.data.frame(surv.NPM1.2)
autoplot(surv.NPM1.2)
names(mod.Lgn)
