using HDF5

struct GTEXData 
    data::Array{Float64, 2}
    cols::Array{String, 1}
    rows::Array{String, 1}
end 
function get_GTEX_data()
    inf = h5open("Data/GTEX/GTEX.out", "r")
    tpm = log10.(inf["data"][:,:] .+ 1) 
    rows = inf["rows"][:]
    cols = inf["cols"][:]
    tissues = inf["tissues"][:]
    close(inf)
    gtexd = GTEXData(tpm, cols, rows)
    return gtexd, tissues 
end 
