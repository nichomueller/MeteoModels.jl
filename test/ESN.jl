using MeteoModels
using OrdinaryDiffEq
using ReservoirComputing
using Random
using Test
using LinearAlgebra
using Gridap
using Gridap.Arrays

# function lorenz!(du,u,p,t)
#     du[1] = 10.0 * (u[2] - u[1])
#     du[2] = u[1] * (28.0 - u[3]) - u[2]
#     du[3] = u[1] * u[2] - (8 / 3) * u[3]
# end
function lorenz!(du,u,p,t)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    du[3] = u[1] * u[2] - p[3] * u[3]
end

prob = ODEProblem(lorenz!,[1.0,0.0,0.0],(0.0,200.0),(10.0,28.0,8/3))
data = OrdinaryDiffEq.solve(prob,ABM54(); dt=0.02)
data = reduce(hcat,data.u)

shift = 300
train_len = 5000
predict_len = 1250

input_data = data[:,shift:(shift + train_len - 1)]
target_data = data[:,(shift + 1):(shift + train_len)]
test_data = data[:,(shift + train_len + 1):(shift + train_len + predict_len)]

nstate = 300
ninput = 3
radius = 1
sparsity = 6 / 300
in_scaling = 0.1

rng = MersenneTwister(1234)

esn = EchoStateNetwork(
    ninput,nstate,ninput;
    rng,
    radius,
    sparsity,
    scaling=in_scaling,
    modifier_in=DoNotModify(),
    modifier_state=DoNotModify(),
    activation=tanh
)

net = ESN(ninput,nstate,ninput; 
    init_reservoir = rand_sparse(Float64; radius,sparsity),
    init_input = weighted_init(Float64; scaling = in_scaling),
    init_state = zeros64
)

ps,st = setup(rng,net)

# use the same weights
copyto!(esn.weights,ps.reservoir.reservoir_matrix)
copyto!(esn.weights_in,ps.reservoir.input_matrix)

@test esn.weights ≈ ps.reservoir.reservoir_matrix 
@test esn.weights_in ≈ ps.reservoir.input_matrix 

method = TrainRecurrentNeuralNetwork(
    augmentation=NoAugmentation(),
    regularisation= NoRegularisation(),
    λ=1e-6
)

states = train(method,esn,input_data,target_data)
ps,st = train!(net,input_data,target_data,ps,st,StandardRidge(1e-6))

@test st.states ≈ states 
@test ps.readout.weight ≈ esn.weights_out_T'

y = forecast(esn,test_data[:,1],1:predict_len)
output,st = predict(net,predict_len,ps,st; initialdata=test_data[:,1])

# this is different due to round-off,nothing to worry about I think
# @test output ≈ y 

# recycle validation

Nfolds = 4
tfold = 20
δ = floor(Int,train_len / Nfolds)
starts = [δ*(i-1) + 1 for i = 1:Nfolds]
windows = [start:start+tfold-1 for start in starts]

radii = 0.8:0.1:1.0
in_scalings = 0.1:0.1:0.3

updates = map(radii,in_scalings) do radius,scaling
    (
        rand_sparse(rng,Float64,nstate,nstate;radius,sparsity),
        weighted_init(rng,Float64,nstate,ninput;scaling)
    )
end

rvmethod = RecycleValidation(method,updates,windows)
rvstates = train(rvmethod,esn,input_data,target_data)

# jacobian 

x = data[:,end]
J = jac(esn,x)

o = ones(nstate)
T = o - (tanh.(esn.weights_in * x + esn.weights * esn.state)).^2
TT = stack([T for _ = 1:ninput])
Jtest = -esn.weights_out_T'*(TT .* esn.weights_in)

@test J ≈ Jtest 

# now with normalization

esn_norm = EchoStateNetwork(
    ninput,nstate,ninput;
    rng,
    radius,
    sparsity,
    scaling=in_scaling,
    activation=tanh
)

train(method,esn_norm,input_data,target_data)

g = 1 ./ esn_norm.modifier_in.factor
J = jac(esn_norm,x)

T = o - (tanh.(esn_norm.weights_in * vcat(x .* g,esn_norm.modifier_in.value) + esn_norm.weights * esn_norm.state)).^2
TT = stack([T for _ = 1:ninput])
GG = vcat([g' for _ = 1:nstate]...)
Jtest = -esn_norm.weights_out_T[1:end-1,:]'*(TT .* (esn_norm.weights_in[:,1:end-1] .* GG))

@test J ≈ Jtest 

