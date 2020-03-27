function w_out = train(params, states, data)

[statesize, ~] = size(states);

beta = params.beta;

idenmat = beta*speye(statesize);

w_out = data*transpose(states)*pinv(states*transpose(states)+idenmat);