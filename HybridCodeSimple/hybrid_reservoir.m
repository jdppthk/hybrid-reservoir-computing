%start
clear;

%generate noisy lorenz data

noise_scale = 1;
ModelParams.a = 10;
ModelParams.b = 28;
ModelParams.c = 8/3;
ModelParams.tau = 0.1;
ModelParams.nstep = 10000;
assimilation_interval = 10;
rng('shuffle')
init_cond = rand(1,3);
data = transpose(rk_lorenz_solve(init_cond,ModelParams));

measured_vars = [1,2,3];
num_measured = length(measured_vars);
R = noise_scale*eye(num_measured);
mu = zeros(1,num_measured);
%z = transpose(mvnrnd(mu,R,length(data)));
measurements = data(measured_vars, :) ;%+ z;


%train reservoir
[num_inputs,~] = size(measurements);
resparams.radius = 0.4;
resparams.degree = 3;
resparams.N = 500;
resparams.sigma = 0.1;
resparams.train_length = 5000;
resparams.num_inputs = num_inputs;
resparams.predict_length = 500;
resparams.beta = 0.0001;

ModelParams.a = 10;
ModelParams.b = 28+0.01*28;
ModelParams.c = 8/3;
ModelParams.tau = 0.1;
ModelParams.nstep = 1;

model_predictions = zeros(num_inputs, resparams.train_length);

model_predictions(:,1) = measurements(:,1);

for i = 1:resparams.train_length
    model_predictions(:,i+1) = rk4(@f, measurements(:,i), 0, ModelParams);
end

A = generate_reservoir(resparams.N, resparams.radius, resparams.degree);

input = measurements(:, 1:resparams.train_length);

[input_size, ~] = size(input); 

win = resparams.sigma*(-1 + 2*rand(resparams.N, input_size));

states = reservoir_layer(A, win, input, resparams);

augmented_states = vertcat(model_predictions(:,2:end), states(:,2:end));

wout = train(resparams, augmented_states, measurements(:,2:resparams.train_length+1));
x = states(:,end);

output = zeros(resparams.num_inputs, resparams.predict_length);


y = model_predictions(:, end);
out = wout*(vertcat(y,x));
output(:,1) = out;
for i = 1:resparams.predict_length
    x = tanh(A*x + win*out);
    y = rk4(@f, out, 0, ModelParams);
    out = wout*(vertcat(y,x));
    output(:,i+1) = out;
end


plot(output(1,:))
hold on
plot(measurements(1, resparams.train_length+1 : resparams.train_length + resparams.predict_length))

figure()
plot(wout(1,:))