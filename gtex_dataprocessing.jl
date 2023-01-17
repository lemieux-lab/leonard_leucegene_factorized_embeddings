using HDF5

struct GTEXData 
    data::Array{Float64, 2}
    cols::Array{String31, 1}
    rows::Array{String31, 1}
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
