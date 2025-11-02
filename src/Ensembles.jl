abstract type Ensemble <: Iterables end

get_data(e::Ensemble) = @abstractmethod
get_state(e::Ensemble) = get_data(e)
Statistics.mean(e::Ensemble) = mean(get_data(e),dims=2)
Statistics.cov(e::Ensemble) = cov(get_data(e))

struct EnsembleOperators{A} <: Operators 
  op::A
  ensemble_size::Int
end

state_size(op::EnsembleOperators) = state_size(op.op)
measurement_size(op::EnsembleOperators) = measurement_size(op.op)
ensemble_size(op::EnsembleOperators) = op.ensemble_size

function allocate_iterables(op::EnsembleOperators;kwargs...)
  @abstractmethod
end

function update!(op::EnsembleOperators,args...)
  update!(op.op,args...)
end

abstract type EnsembleCache <: FilterCache end

const EnsembleFilter{A<:EnsembleOperators,B<:Ensemble,C<:EnsembleCache} = Filter{A,B,C}