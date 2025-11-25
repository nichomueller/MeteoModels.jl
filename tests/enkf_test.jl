using MeteoModels
using LinearAlgebra
using Statistics
using Plots

# Lorenz-96 model
function lorenz96!(dx,x,f)
  n = length(x)
  @inbounds for i in 1:n
    dx[i] = (x[mod1(i+1,n)] - x[mod1(i-2,n)]) * x[mod1(i-1,n)] - x[i] + f
  end
  return dx
end

function step_l96!(x,dt,f)
  dx = similar(x)
  lorenz96!(dx,x,f)
  @. x += dt * dx
  return x
end

n = 40          
ne = 40
F = 8.0
dt = 0.01
Nt = 100

# Initial ensemble with small random perturbations
X = F .+ 0.1 * randn(n,ne)

# Observation operator (observe every 2nd variable)
no = n ÷ 2
H = zeros(Int,no,n)
for i in 1:no
  H[i,2*i-1] = 1
end

Q = 0.5 * I(n)
R = 0.5 * I(no) 

iter = KalmanEnsemble(X)
op = EnKFOperators(I(n),H,Q,R;ensemble_size=ne)
kf = Filter(op,iter)
xens = zeros(eltype(X),size(X)...,Nt)

# truth 
xtrue = F .+ 0.1 * randn(n)
xtrueall = zeros(eltype(xtrue),size(xtrue)...,Nt)

for k in 1:Nt   
  step_l96!(xtrue,dt,F) 
  for j in 1:ne
    step_l96!(X[:,j],dt,F)
  end
  y = Observation(k*dt,H * xtrue + 0.1*randn(no))
  kf(y)
  @views xtrueall[:,k] = copy(xtrue)
  @views xens[:,:,k] = MeteoModels.get_state(kf)
end

# compute ensemble spread (std deviation)
xstd = [std(xens[i,:,t]) for i in axes(xens,1),t in axes(xens,3)]

# Example: plot first state variable
i = 1
times = collect(dt:dt:Nt*dt)
xmean = dropdims(mean(xens,dims=2);dims=2)
plot(times,xtrueall[i,:],color=:black,label="True state")
plot!(times,xmean[i,:],color=:blue,label="EnKF mean")
plot!(times,xmean[i,:] .+ xstd[i,:],color=:blue,linestyle=:dash,label="±1 std")
plot!(times,xmean[i,:] .- xstd[i,:],color=:blue,linestyle=:dash,label="")
xlabel!("Time step")
ylabel!("x[$i]")
title!("State evolution for variable $i")

# plot(times,xens[i,1,:])

k = 1
step_l96!(xtrue,dt,F) 
for j in 1:ne
  step_l96!(X[:,j],dt,F)
end
y = Observation(k*dt,H * xtrue + 0.1*randn(no))