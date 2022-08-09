# inference tests 
using Flux 
using RedefStructs

@redef struct NN{A, B}
    layer1::Flux.Embedding{A}
    layer2::Flux.Embedding{A}
    net::Flux.Chain{B}

end 
l1 = Flux.Embedding(300,10)
l2 = Flux.Embedding(5000,10)
net = Flux.Chain(Flux.Parallel(vcat, l1, l2), Flux.Dense(20, 10, relu), Flux.Dense(10,1, identity), vec)
nn = NN(l1, l2, net)

# split data 
# train FE  
#  