using LinearAlgebra
using Gridap
using GridapROMs
using MeteoModels
using ChainRulesCore
using Zygote
using NLopt

using GridapROMs.RBSteady

method=:pod
compression=:global
hypred_strategy=:mdeim
tol=1e-4
nparams=50
nparams_res=floor(Int,nparams/3)
nparams_jac=floor(Int,nparams/4)
sketch=:sprn
ncentroids=2

pdomain = (1,10,1,10,1,10)
pspace = ParamSpace(pdomain)

domain = (0,1,0,1)
partition = (10,10)
if method==:ttsvd
  model = TProductDiscreteModel(domain,partition)
else
  model = CartesianDiscreteModel(domain,partition)
end

order = 2
degree = 2*order

Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model,tags=[8])
dΓn = Measure(Γn,degree)

a(μ) = x -> exp(-x[1]/sum(μ))
aμ(μ) = parameterise(a,μ)

f(μ) = x -> 1.
fμ(μ) = parameterise(f,μ)

g(μ) = x -> μ[1]*exp(-x[1]/μ[2])
gμ(μ) = parameterise(g,μ)

h(μ) = x -> abs(cos(μ[3]*x[2]))
hμ(μ) = parameterise(h,μ)

stiffness(μ,u,v,dΩ) = ∫(aμ(μ)*∇(v)⋅∇(u))dΩ
rhs(μ,v,dΩ,dΓn) = ∫(fμ(μ)*v)dΩ + ∫(hμ(μ)*v)dΓn
res(μ,u,v,dΩ,dΓn) = stiffness(μ,u,v,dΩ) - rhs(μ,v,dΩ,dΓn)

trian_res = (Ω,Γn)
trian_stiffness = (Ω,)
domains = FEDomains(trian_res,trian_stiffness)

energy(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
trial = ParamTrialFESpace(test,gμ)
X = assemble_matrix(energy,trial,test)
C = cholesky(X)

if method == :pod
  state_reduction = Reduction(tol,energy;nparams,sketch,compression,ncentroids)
elseif method == :ttsvd
  state_reduction = Reduction(fill(tol,3),energy;nparams,sketch,compression,ncentroids)
end

fesolver = LUSolver()
rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac,hypred_strategy)

feop = LinearParamOperator(res,stiffness,pspace,trial,test,domains)

nu = num_free_dofs(test)
np = param_dimension(pspace)
δ = 1
stencil = 1:δ:nu
nobs = length(stencil)
R = 0.5^2 * Float64.(I(nobs))
obs_noise = Noise(R)
observationf(u) = u[stencil]
observation = Model(observationf)

μtrue = realisation(pspace,sampling=:uniform)
xtrue, = solution_snapshots(rbsolver,feop,μtrue)
true_p = μtrue.params[1]
true_u = xtrue[:,1]
true_obs = observationf(true_u) + draw(obs_noise)

α = 1
β = 1

function state_value(p)
  μ = RBSteady.to_realisation(μtrue,p) 
  sμ, = solve(fesolver,feop,μ)
  sμ[1]
end 

function loss(p)
  μ = RBSteady.to_realisation(μtrue,p) 
  Uμ = trial(μ)
  sμ, = solve(fesolver,feop,μ)
  uμ = FEFunction(Uμ,sμ)
  rμ = assemble_vector(v->res(μ,uμ,v,dΩ,dΓn),Uμ)[1] 
  oμ = observationf(sμ[1])

  z = similar(rμ)
  ldiv!(z,C,rμ)
  dμ = true_obs - oμ
  (α*rμ'*z + β*dμ'*dμ) / 2
end 

function loss(p,grad)
  if length(grad) > 0
    ∂up, = Zygote.gradient(loss,p)
    copyto!(grad,∂up)
  end
  loss(p)
end

function optimise_loss(p;tol=1e-4,maxiter=20)
  opt = Opt(:LD_MMA,np)
  opt.lower_bounds = [1.,1.,1.]
  opt.upper_bounds = [10.,10.,10.]
  opt.ftol_rel = tol
  opt.maxeval = maxiter
  opt.max_objective = loss

  opt_state,opt_p, = NLopt.optimize(opt,p)
  opt_μ = RBSteady.to_realisation(μtrue,opt_p)
  @show numevals = opt.numevals 

  opt_state,opt_μ
end

p0 = [5.,5.,5.]
# opt_state,opt_μ = optimise_loss(p0)

function state_value_gridap(p)
  μ = RBSteady.to_realisation(μtrue,p) 
  af(u,v) = ∫(a(p)*∇(v)⋅∇(u))dΩ
  lf(v) = ∫(f(p)*v)dΩ + ∫(h(p)*v)dΓn
  U = param_getindex(trial(μ),1)
  op = AffineFEOperator(af,lf,U,test)
  solve(op)
end 

function _gridap_operator_state(p)
  μ = RBSteady.to_realisation(μtrue,p)
  af(u,v) = ∫(a(p)*∇(v)⋅∇(u))dΩ
  lf(v) = ∫(f(p)*v)dΩ + ∫(h(p)*v)dΓn
  U = param_getindex(trial(μ),1)
  op = AffineFEOperator(af,lf,U,test)
  uh = solve(op)
  return op,U,uh
end

function _gridap_residual_vector(p,uh)
  res(v) = ∫(a(p)*∇(v)⋅∇(uh))dΩ - ∫(f(p)*v)dΩ - ∫(h(p)*v)dΓn
  assemble_vector(res,test)
end

function _a_coeff(p)
  s = sum(p)
  x -> exp(-x[1] / s)
end

function _da_dp_coeff(p,i)
  s = sum(p)
  if i == 1 || i == 2
    x -> exp(-x[1] / s) * x[1] / (s^2)
  elseif i == 3
    x -> 0.0
  else
    error("Unsupported parameter index $i")
  end
end

function _dh_dp3_coeff(p)
  x -> begin
    c = cos(p[3] * x[2])
    s = sin(p[3] * x[2])
    # Subgradient at c == 0 is set to 0 for numerical robustness.
    ifelse(abs(c) < 1e-12, 0.0, -sign(c) * s * x[2])
  end
end

function _dR_dp_vector(p,uh,i)
  if i == 1 || i == 2
    return assemble_vector(v -> ∫(_da_dp_coeff(p,i) * ∇(v)⋅∇(uh))dΩ,test)
  elseif i == 3
    dhneg(x) = -_dh_dp3_coeff(p)(x)
    return assemble_vector(v -> ∫(dhneg * v)dΓn,test)
  else
    error("Unsupported parameter index $i")
  end
end

function _dJ_du_vector(p,op,uh)
  u = get_free_dof_values(uh)
  r = _gridap_residual_vector(p,uh)
  z = similar(r)
  ldiv!(z,C,r)
  d = true_obs - observationf(u)

  A = Gridap.jacobian(op,uh)
  state_term = α .* (A' * z)

  obs_term = zeros(eltype(state_term),length(state_term))
  obs_term[stencil] .= -β .* d

  state_term + obs_term
end

function loss_gridap(p)
  uh = state_value_gridap(p)
  u = get_free_dof_values(uh)
  r = _gridap_residual_vector(p,uh)
  o = observationf(u)

  z = similar(r)
  ldiv!(z,C,r)
  d = true_obs - o
  (α*r'*z + β*d'*d) / 2
end 

function adjoint_gradient_gridap(p)
  op,_,uh = _gridap_operator_state(p)

  dJdu = _dJ_du_vector(p,op,uh)
  A = Gridap.jacobian(op,uh)
  λadj = A' \ dJdu

  r = _gridap_residual_vector(p,uh)
  z = similar(r)
  ldiv!(z,C,r)

  g = zeros(length(p))
  for i in eachindex(p)
    dRdpi = _dR_dp_vector(p,uh,i)
    g[i] = α * dot(z,dRdpi) - dot(λadj,dRdpi)
  end
  g
end

function loss_gridap(p,grad)
  if length(grad) > 0
    copyto!(grad,adjoint_gradient_gridap(p))
  end
  loss_gridap(p)
end

function ChainRulesCore.rrule(::typeof(loss_gridap),p::AbstractVector{<:Real})
  y = loss_gridap(p)
  g = adjoint_gradient_gridap(collect(Float64.(p)))
  function loss_gridap_pullback(ȳ)
    Δ = ChainRulesCore.unthunk(ȳ)
    return NoTangent(),Δ.*g
  end
  return y,loss_gridap_pullback
end

Zygote.gradient(loss_gridap,p0)
