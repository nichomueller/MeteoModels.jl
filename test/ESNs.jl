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
arr = stack([mat,mat]) 
reg0 = NoRegularisation()
reg = DataRegularisation(arr)
@test evaluate(reg,mat) != evaluate(reg0,mat) == mat
aug = DataAugmentation(-1)
augmat = evaluate(aug,mat)
@test augmat[:,1,:] == mat 
@test augmat[:,2,:] == -mat

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
radius = 0.9
connect = 5
scaling = 0.1

esn = EchoStateNetwork(
    ninput,nstate,ninput;
    radius,
    connect,
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
    states[:,i] = tanh.(esn.scaling[] .* (esn.weights_in * input_data[:,i]) .+ esn.radius[] .* (esn.weights * x))
    copyto!(x,states[:,i])
end
wstates = MeteoModels.apply_washout(states,washout)
rhs = target_data[:,washout+1:end]
lhs = wstates
LHS = lhs * lhs' + λ * I(size(lhs,1))
RHS = lhs * rhs'
_LHS = copy(LHS)
C = cholesky!(_LHS)
weights_out = zeros(size(lhs,1),size(rhs,1))
ldiv!(weights_out,C,RHS)

esn_states = train(method,esn,input_data,target_data)

@test wstates ≈ esn_states 
@test norm(weights_out - esn.weights_out_T) / norm(weights_out) < 1e-4
copyto!(esn.weights_out_T,weights_out)

reset_state!(esn)
y = evaluate(esn,test_data[:,1],1:predict_len)
@test y[:,1] == test_data[:,1]

function test_forecast(inp)
    x = zeros(length(esn.state))
    for i in 2:predict_len
        x = tanh.(esn.scaling[] .* (esn.weights_in * inp) .+ esn.radius[] .* (esn.weights * x))
        inp = esn.weights_out_T' * x
        @test norm(y[:,i] - inp) / norm(inp) < 1e-10
    end
end
test_forecast(y[:,1])

# recycle validation

Nfolds = 4
Ntrain = train_len
Nvalidation = 20

# radius_ranges = 0.8:0.1:1.0
# scaling_ranges = 0.1:0.1:0.3
radius_ranges = range(0.7,1.05,length=4)
scaling_ranges = range(1e-5,1.0,length=4) 
# radius_ranges = range(0.7,1.05,length=4)
# scaling_ranges = range(LogNumber{10}(log10(1e-5)),LogNumber{10}(log10(1.0)),length=4)

rvmethod = RecycleValidation(method,radius_ranges,scaling_ranges;Nfolds,Ntrain,Nvalidation)
train(rvmethod,esn,input_data,target_data)

tikhonov = (1e-16,1e-12,1e-10,1e-8)
rvmethod_tikhonov = RecycleValidation(method,tikhonov,radius_ranges,scaling_ranges;Nfolds,Ntrain,Nvalidation)
train(rvmethod_tikhonov,esn,input_data,target_data)

# jacobian 

x = data[:,end]
J = jac(esn,x)

o = ones(nstate)
T = o - (tanh.(esn.scaling[] .* (esn.weights_in * x) .+ esn.radius[] .* (esn.weights * esn.state))).^2
TT = stack([T for _ = 1:ninput])
Jtest = esn.scaling[] .* (esn.weights_out_T' * (TT .* esn.weights_in))

@test J ≈ Jtest 

# now with modifiers

# esn = EchoStateNetwork(
#     ninput,nstate,ninput;
#     radius,
#     connect,
#     scaling,
#     modifier_in=Modifier(Normalisation(ones(ninput)),NoTransformation(),AddBias(1.0)),
#     modifier_state=Modifier(NoNormalisation(),T₂(),AddBias(0.1)),
#     activation=tanh
# )
esn = EchoStateNetwork(
    ninput,nstate,ninput;
    radius,
    connect,
    scaling,
    modifier_in=Modifier(Normalisation(ones(ninput)),NoTransformation(),AddBias(1.0)),
    modifier_state=Modifier(NoNormalisation(),T₂(),AddBias(0.1)),
    activation=tanh
)

train(method,esn,input_data,target_data)

g = 1 ./ esn.modifier_in.normalisation.factor
J = jac(esn,x)

S = tanh.(esn.scaling[] .* (esn.weights_in * vcat(x .* g,esn.modifier_in.bias.value)) .+ esn.radius[] .* (esn.weights * esn.state))
T = o - S.^2
TT = stack([T for _ = 1:ninput])
GG = vcat([g' for _ = 1:nstate]...)
Js = jac(esn.modifier_state,S)[1:end-1,:]
Jtest = esn.scaling[] .* (esn.weights_out_T[1:end-1,:]'*Js*(TT .* (esn.weights_in[:,1:end-1] .* GG)))

@test J ≈ Jtest 

# performance
using Plots

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
