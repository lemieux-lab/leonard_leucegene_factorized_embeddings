using CairoMakie
using AlgebraOfGraphics
using DataFrames
using Flux
set_aog_theme!()

# df = DataFrame(patient_embed_mat[:,:], :auto)
# p = data(df) * mapping(:x1) * mapping(:x2)
# draw(p)

# include("data_preprocessing.jl")
# X_, Y_ = DataPreprocessing.prep_data(ge_cds_all)
y = cpu(model.net(X))

df = DataFrame(y = y, Y = cpu(Y))
p = data(df) * mapping(:y, :Y) * AlgebraOfGraphics.density(npoints=100)
draw(p * visual(Heatmap), ; axis=(width=1024, height=1024))

