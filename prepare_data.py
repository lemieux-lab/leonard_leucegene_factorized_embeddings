from base_datasets import * 
# 1) create Survival and Gene Expression Dataset object
SGE = SurvivalGEDataset()
my_data = SGE.get_data("lgn_pronostic")
pdb.set_trace()