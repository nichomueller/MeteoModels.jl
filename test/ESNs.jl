module ESNTest
   
using MeteoModels
using OrdinaryDiffEq
using Test
using LinearAlgebra
using Gridap
using Gridap.Arrays

n = 3
x = rand(n)
a = 2
factor = rand(n)
transformation = T₂()
normalisation = Normalisation(factor)
bias = AddBias(a)
modifier = Modifier(normalisation,transformation,bias)
x1 = x ./ factor 
x2 = [x1[1],x1[2],x1[1]*x1[2]]
x3 = vcat(x2,a)
@test evaluate(modifier,x) == x3 
J1 = jac(normalisation,x)
@test J1 == diagm(1 ./ factor)
J2 = jac(transformation,x1)
@test J2 == [
    1 0 0 
    0 1 0 
    x1[2] x1[1] 0    
]
J3 = jac(bias,x2)
@test J3 == [
    1 0 0 
    0 1 0 
    0 0 1
    0 0 0    
]
@test jac(modifier,x) == J3*J2*J1

mat = rand(n,n)
arr = stack([mat,mat]) # std is zero,so no regularisation 
reg0 = NoRegularisation()
reg = DataRegularisation(arr)
@test evaluate(reg,mat) == evaluate(reg0,mat) == mat
aug = DataAugmentation(-1)
@test evaluate(aug,mat) == hcat(mat,-mat)

function lorenz!(du,u,p,t)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    du[3] = u[1] * u[2] - p[3] * u[3]
end

dt = 0.01
prob = ODEProblem(lorenz!,[1.0,0.0,0.0],(0.0,200.0),(10.0,28.0,8/3))
data = OrdinaryDiffEq.solve(prob,Tsit5();dt,saveat=dt:dt:200.0)
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
scaling = 0.1

esn = EchoStateNetwork(
    ninput,nstate,ninput;
    radius,
    sparsity,
    scaling,
    modifier_in=DoNotModify(),
    modifier_state=DoNotModify(),
    activation=tanh
)

washout = 30
λ = 1e-6
method = TrainRecurrentNeuralNetwork(
    augmentation=NoAugmentation(),
    regularisation= NoRegularisation(),
    washout=washout,λ=λ
)

states = zeros(length(esn.state),size(input_data,2)) 
x = copy(esn.state)
for i in axes(states,2)
    states[:,i] = tanh.(esn.weights_in * input_data[:,i] + esn.weights * x)
    copyto!(x,states[:,i])
end
rhs = target_data[:,washout+1:end]
lhs = states[:,washout+1:end]
LHS = vcat(lhs',(sqrt(λ) * I(size(lhs,1))))
RHS = vcat(rhs',zeros(size(lhs,1),size(rhs,1)))
weights_out = qr(LHS) \ RHS  

esn_states = train(method,esn,input_data,target_data)

@test states ≈ esn_states 
@test weights_out ≈ esn.weights_out_T

MeteoModels.reset_state!(esn)
y = evaluate(esn,test_data[:,1],1:predict_len)
@test y[:,1] == test_data[:,1]

function test_forecast(inp)
    x = zeros(length(esn.state))
    for i in 2:predict_len
        x = tanh.(esn.weights_in * inp + esn.weights * x)
        inp = esn.weights_out_T' * x 
        @test norm(y[:,i] - inp) / norm(inp) < 1e-6
    end
end
test_forecast(y[:,1])

# recycle validation

Nfolds = 4
tfold = 20
δ = floor(Int,train_len / Nfolds)
starts = [δ*(i-1) + 1 for i = 1:Nfolds]
windows = [start:start+tfold-1 for start in starts]

radius = 0.8:0.1:1.0
scaling = 0.1:0.1:0.3

rvmethod = RecycleValidation(method,ninput,nstate,windows;radius,scaling,sparsity)
rvstates = train(rvmethod,esn,input_data,target_data)

# jacobian 

x = data[:,end]
J = jac(esn,x)

o = ones(nstate)
T = o - (tanh.(esn.weights_in * x + esn.weights * esn.state)).^2
TT = stack([T for _ = 1:ninput])
Jtest = esn.weights_out_T'*(TT .* esn.weights_in)

@test J ≈ Jtest 

# now with modifiers

esn = EchoStateNetwork(
    ninput,nstate,ninput;
    radius=radius[1],
    sparsity,
    scaling=scaling[1],
    modifier_in=Modifier(Normalisation(ones(ninput)),NoTransformation(),AddBias(1.0)),
    modifier_state=Modifier(NoNormalisation(),T₂(),AddBias(1.0)),
    activation=tanh
)

train(method,esn,input_data,target_data)

g = 1 ./ esn.modifier_in.normalisation.factor
J = jac(esn,x)

S = tanh.(esn.weights_in * vcat(x .* g,esn.modifier_in.bias.value) + esn.weights * esn.state)
T = o - S.^2
TT = stack([T for _ = 1:ninput])
GG = vcat([g' for _ = 1:nstate]...)
Js = jac(esn.modifier_state,S)[1:end-1,:]
Jtest = esn.weights_out_T[1:end-1,:]'*Js*(TT .* (esn.weights_in[:,1:end-1] .* GG))

@test J ≈ Jtest 

# performance
using Plots, Plots.PlotMeasures

ts = 0.0:dt:200.0
lorenz_maxlyap = 0.9056
predict_ts = ts[(shift + train_len + 1):(shift + train_len + predict_len)]
lyap_time = (predict_ts .- predict_ts[1]) * (1 / lorenz_maxlyap)

p1 = plot(lyap_time,[test_data[1,:] y[1,:]]; label=["actual" "predicted"],
    ylabel="x(t)",linewidth=2.5,xticks=false,yticks=-15:15:15);
p2 = plot(lyap_time,[test_data[2,:] y[2,:]]; label=["actual" "predicted"],
    ylabel="y(t)",linewidth=2.5,xticks=false,yticks=-20:20:20);
p3 = plot(lyap_time,[test_data[3,:] y[3,:]]; label=["actual" "predicted"],
    ylabel="z(t)",linewidth=2.5,xlabel="max(λ)*t",yticks=10:15:40);

plot(p1,p2,p3; plot_title="Lorenz System Coordinates",
    layout=(3,1),xtickfontsize=12,ytickfontsize=12,xguidefontsize=15,
    yguidefontsize=15,
    legendfontsize=12,titlefontsize=20)

end