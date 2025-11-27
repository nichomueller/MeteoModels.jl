const ExtendedKalmanOperators{A<:GenericModel,B<:GenericModel,C,D} = KalmanOperators{A,B,C,D}

function predict!(i::KalmanIterables,cache::KalmanCache,op::ExtendedKalmanOperators,x::Observation)
  trans_model = discretize(op.trans_model,i)
  obser_model = discretize(op.trans_model,nothing)
  algop = KalmanOperators(trans_model,obser_model,op.proce_noise,op.obser_noise)
  predict!(i,cache,algop,x)
  return i
end

function update!(i::KalmanIterables,cache::KalmanCache,op::ExtendedKalmanOperators,x::Observation)
  trans_model = discretize(op.trans_model,nothing)
  obser_model = discretize(op.trans_model,i)
  algop = KalmanOperators(trans_model,obser_model,op.proce_noise,op.obser_noise)
  predict!(i,cache,algop,x)
  return i
end

const ExtendedKalmanFilter{A<:ExtendedKalmanOperators,B<:KalmanIterables,C<:KalmanCache} = Filter{A,B,C}
