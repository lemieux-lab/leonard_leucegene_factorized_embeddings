basepath = "/u/sauves/public_html/LEUCEGENE/RES2022_EMBEDDINGS/embeddings_2023-03-27T14:20:42.846/FE_45b5a76efe112fbdf31e0"
f = readlines("$basepath/training_curves.txt")
FE_acc = []
FE_loss = []
CLF_acc = []
CLF_loss = []

for line in f
    lm = split(line, ",")
    fe_loss = parse(Float32, strip(split(lm[2], ":")[2]))
    fe_acc = parse(Float32, strip(split(lm[3], ":")[2]))
    clf_loss = parse(Float32, strip(split(lm[4], ":")[2]))
    clf_acc = parse(Float32, strip(split(lm[5], ":")[2][1:end-1]))
    
    push!(FE_acc, fe_acc)
    push!(FE_loss, fe_loss)
    push!(CLF_acc, clf_acc)
    push!(CLF_loss, clf_loss)
end 

my_data = DataFrame(Dict("step"=>collect(1:length(FE_acc)), "FE_acc"=>FE_acc,  "CLF_acc"=>CLF_acc))


my_data_loss = DataFrame(Dict("step"=>collect(1:length(FE_acc)), "FE_loss"=>FE_loss,  "CLF_loss"=>CLF_loss))

p = data(my_data) * mapping(:step, :CLF_acc) * visual(Lines, color="blue")
g = data(my_data) * mapping(:step, :FE_acc) * visual(Lines, color="red")
fig1 = draw(p + g, axis = (;title = "Training Curve for Dual Objective FE model by gradient step \nRED: FE pearson, BLUE: CLF accuracy", ylabel="Accuracy"))
CairoMakie.save("$(basepath)_tr_curves_acc.png", fig1)    


h = data(my_data_loss) * mapping(:step, :CLF_loss) * visual(Lines, color="blue")
fig2 = draw(h, axis = (;title = "Training Curve for Dual Objective FE model by gradient step \nRED: FE loss (MSE), BLUE: CLF loss (Crossentropy)", ylabel="MSE / Crossentropy Loss"))
CairoMakie.save("/u/sauves/public_html/LEUCEGENE/RES2022_EMBEDDINGS/embeddings_2023-03-27T14:20:42.846/FE_6a736d1303185737082f2_tr_curves_CLF_loss.png", fig2)

g = data(my_data_loss) * mapping(:step, :FE_loss) * visual(Lines, color="red")
fig3 = draw(g, axis = (;title = "Training Curve for Dual Objective FE model by gradient step \nRED: FE loss (MSE), BLUE: CLF loss (Crossentropy)", ylabel="MSE / Crossentropy Loss"))
CairoMakie.save("/u/sauves/public_html/LEUCEGENE/RES2022_EMBEDDINGS/embeddings_2023-03-27T14:20:42.846/FE_6a736d1303185737082f2_tr_curves_FE_loss.png", fig3)
CLF_loss