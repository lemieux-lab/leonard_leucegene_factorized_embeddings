using Pkg
Pkg.activate(".")
using CairoMakie
using AlgebraOfGraphics
using DataFrames
using ProgressBars
#### CairoMakie (without algebra of graphics)
#### Data
infile = readlines("tmp.out") 
metrics = Array{Float32, 2}(undef, (length(infile), 4))
infile[1]
for line in ProgressBar(infile)
    lsplit  = split(line, ",")
    itn = parse( Int, lsplit[1])
    metrics[itn, :] = [parse(Base.Float32, split(x, ":")[2][1:end-1]) for x in  lsplit[2:end]]
end 
#### Scatterplot
test_range = collect(1:size(metrics)[1])
f = Figure()
ax = Axis(f[1,1])
lines!(ax, collect(1:size(metrics)[1])[test_range] ,metrics[test_range,2])
lines!(ax, collect(1:size(metrics)[1])[test_range] ,metrics[test_range,4])

save("cairo_makie_test.svg", f)
save("cairo_makie_test.png", f) 