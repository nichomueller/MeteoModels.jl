const StackedMatrix = BlockMatrix 

function stack_matrices(A::AbstractVector{<:AbstractMatrix})
  mortar(reshape(A,length(A),1))
end

