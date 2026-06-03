using LinearAlgebra
using Gridap
using GridapROMs
using MeteoModels
using ChainRulesCore
using Zygote
using NLopt
using ReverseDiff

using GridapROMs.RBSteady
using GridapROMs.ParamDataStructures

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
H = zeros(nobs,nu)
for (io,i) in enumerate(stencil)
  H[io,i] = 1.0
end
observation = Model(H)

μtrue = realisation(pspace,sampling=:uniform)
xtrue, = solution_snapshots(rbsolver,feop,μtrue)
true_p = μtrue.params[1]
true_u = xtrue[:,1]
true_obs = observation(true_u) + draw(obs_noise)

# function gridap_jac(p)
#   af(u,v) = ∫(a(p) * ∇(v) ⋅ ∇(u))dΩ
#   μ = RBSteady.to_realisation(μtrue,p)
#   U = param_getindex(trial(μ),1)
#   assemble_matrix(af,U,test)
# end

# function gridap_res(p,uh)
#   af(v) = ∫(a(p) * ∇(v) ⋅ ∇(uh))dΩ
#   lf(v) = ∫(f(p) * v)dΩ + ∫(h(p) * v)dΓn
#   assemble_vector(v -> af(v) - lf(v),test)
# end

# function _vjp_scalarized(p,uh,Δh)
#   term_a = ∫(a(p) * ∇(Δh) ⋅ ∇(uh))dΩ
#   term_f = ∫(f(p) * Δh)dΩ
#   term_h = ∫(h(p) * Δh)dΓn
#   sum(term_a - term_f - term_h)
# end

# function ChainRulesCore.rrule(::typeof(gridap_res),p,uh)
#   primal = gridap_res(p,uh)

#   function gridap_res_pullback(ȳ)
#     Δ = ChainRulesCore.unthunk(ȳ)
#     Δh = FEFunction(test,Δ)
#     pvec = collect(Float64.(p))
#     g = ReverseDiff.gradient(pp -> _vjp_scalarized(pp,uh,Δh),pvec)
#     return ChainRulesCore.NoTangent(),g,ChainRulesCore.NoTangent()
#   end

#   return primal,gridap_res_pullback
# end

# p0 = [5.,5.,5.]
# _,_ ,uh0 = _gridap_operator_state(p0)
# dresdp0, = Zygote.jacobian(gridap_res,p0,uh0)
# Jp0 = gridap_jac(p0)

# A = I(nobs)
# P = R 
# u0 = get_free_dof_values(uh0)
# dloss = -dresdp0' * Jp0' * A' * P * A * (observation(u0) - true_obs)

ad = ADParamIdentification(feop,observation,obs_noise;maxiter=20)
p = identify_parameter(ad,true_obs)

p = MeteoModels.sample_number(ad.op)
u = similar(MeteoModels.get_solution_cache(ad.cache))
fill!(u,zero(eltype(u)))

y = MeteoModels.get_obs_cache(ad.cache)
k = 0
gradnorm = Inf
x = MeteoModels.solve_pde!(ad,p,u)
∂res∂μ = MeteoModels.compute_res_derivative!(ad,p,x)
evaluate!(y,ad.observation,x)
axpy!(-1,y,true_obs)

∂loss∂μ = MeteoModels.compute_loss_derivative!(ad,∂res∂μ,y)

gradnorm = norm(∂loss∂μ)
axpy!(-ad.step_size,∂loss∂μ,p)
copyto!(u,x)
k += 1