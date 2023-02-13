# module Init
using Pkg
if isfile("Project.toml") && isfile("Manifest.toml")
    Pkg.activate(".")
end

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

function set_dirs(basepath)
    outpath  = "./RES/EMBEDDINGS" # our output directory
    outdir = "$(outpath)/embeddings_$(now())"
    mkdir(outdir)
    model_params_list = []
    accuracy_list = []
    return outpath, outdir, model_params_list, accuracy_list
end

# end 