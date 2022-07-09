using Pkg
if isfile("Project.toml") && isfile("Manifest.toml")
    Pkg.activate(".")
end
@time using RedefStructs 
@time using Statistics
@time using CUDA
@time using DataFrames
@time using CSV
@time using Flux
