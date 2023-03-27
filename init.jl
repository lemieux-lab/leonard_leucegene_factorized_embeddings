# module Init
using Pkg
Pkg.activate("/u/sauves/leonard_leucegene_factorized_embeddings/")

# imports  
using CSV
using DataFrames
using JSON
using ProgressBars
using HDF5
using Statistics
using Flux 
using CairoMakie 
using Random
using Dates
using AlgebraOfGraphics
using JuBox
using CUDA

function set_dirs()
    outpath  = "./RES/EMBEDDINGS" # our output directory
    outdir = "$(outpath)/embeddings_$(now())"
    mkdir(outdir)
    model_params_list = []

    return outpath, outdir, model_params_list
end

# end 