function ESN = generate_partzeroed_HESN(NetArgs)
%% Create adjacency matrix for reservoir computer, set nonzero values to be
% random and between -1 and 1, and rescale so that the absolute value of
% the maximum eigenvalue equals rho

RNN_init = sprand(NetArgs.N,NetArgs.N,NetArgs.avg_degree/NetArgs.N);

RNN_init(RNN_init~=0) = 2*RNN_init(RNN_init~=0)-1;

opts.tol = 1e-2;
max_eig = abs(eigs(RNN_init,1,'lr',opts));

A = RNN_init * NetArgs.rho/max_eig;

RNN_input_size = round(NetArgs.N*NetArgs.input_frac);
%% Set input matrix such that each input connects to (on average) N/M nodes
% and set nonzero elements to a uniformly distributed random number in
% [-input_scale input_scale]

% Estimate number of connections per input
input_conn_per_input = floor(RNN_input_size/NetArgs.M);

% Generate input matrix
W_in = zeros(NetArgs.N,2*NetArgs.M);

% Randomly permute the nodes in the reservoir
connections = randperm(NetArgs.N);

%For each input, set it to connect to conn_per_input nodes with a unformly
%distributed random weighting within [-input_scale,input_scale]
for k=1:NetArgs.M
    W_in(connections((k-1)*input_conn_per_input+1:k*input_conn_per_input),k) = ...
        NetArgs.input_scale*(2*rand(input_conn_per_input,1)-1);
end

%For the leftover nodes, generate a random set of inputs and connect these
%to the leftover nodes

num_leftover_nodes = RNN_input_size-input_conn_per_input*NetArgs.M;
leftover_connections = connections((RNN_input_size-num_leftover_nodes+1):RNN_input_size);
leftover_inputs = randperm(NetArgs.M, num_leftover_nodes);

for k=1:num_leftover_nodes
    W_in(leftover_connections(k),leftover_inputs(k)) = NetArgs.input_scale*(2*rand()-1);
end

% Do not assign W_in connections to the model's output, instead leave these
% as zeroes

%% Store reservoir data in a struct
ESN.W_in = W_in;
ESN.A = A;
ESN.r = zeros(NetArgs.N,1);
ESN.bias = NetArgs.bias;
ESN.leakage = NetArgs.leakage;

return