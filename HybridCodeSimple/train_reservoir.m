function [x, wout, A, win] = train_reservoir(resparams, input, target, model_predictions)

A = generate_reservoir(resparams.N, resparams.radius, resparams.degree);

[input_size, ~] = size(input); 

win = resparams.sigma*(-1 + 2*rand(resparams.N, input_size));

states = reservoir_layer(A, win, input, resparams);

size(states)

size(model_predictions)

augmented_states = vertcat(model_predictions, states);

wout = train(resparams, augmented_states, target);

x = states(:,end);

