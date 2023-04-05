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
    session_id = "embeddings_$(now())"
    outpath = "./RES/EMBEDDINGS/$session_id"
    mkdir(outpath)
    model_params_list = []

    return outpath, session_id,  model_params_list
end

# end 