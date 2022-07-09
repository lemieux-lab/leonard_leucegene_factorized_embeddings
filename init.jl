using Pkg
if isfile("Project.toml") && isfile("Manifest.toml")
    Pkg.activate(".")
end
@time using RedefStructs 
@time using Dates
@time using Statistics
@time using CUDA
@time using DataFrames
@time using CSV
@time using Flux

basepath = "/u/sauves/leonard_leucegene_factorized_embeddings/"
outpath  = "./RES/EMBEDDINGS" # our output directory
outdir = "$(outpath)/embeddings_$(now())"
mkdir(outdir)