% kursiv_listen - Driver script for ks predictor
close all;
% Define struct for hyrbrid parameters
NetArgs.N = 500;
NetArgs.bias = 1*(2*rand(NetArgs.N,1)-1);
NetArgs.rho = 0.4;
NetArgs.avg_degree = 3;
NetArgs.input_scale = 1;
NetArgs.leakage = 1;
NetArgs.transient_length = 1000;
NetArgs.training_steps = NetArgs.transient_length + 20000;
NetArgs.moutput_scale = 1;
NetArgs.input_frac = 0.5;
NetArgs.seed = 1;
NetArgs.noise = 0;
NetArgs.listening_steps = 40;
NetArgs.prediction_steps = NetArgs.listening_steps + 2000;
NetArgs.num_predictions = 1;

% Define struct for ks model parameters
ModelParams.d = 35;
ModelParams.N = 64;
ModelParams.tau = .25;
ModelParams.nstep = NetArgs.training_steps + NetArgs.num_predictions*NetArgs.prediction_steps;
ModelParams.model = @solve_kursiv;
ModelParams.const = 0;

NetArgs.M = ModelParams.N;

% Define prediction parameters, if it should plotted, and if the input
% should be fed forward
rmse_max = 0.4;
sys_type = 'hybrid';
plotif = 1;
winput = 0;
N = NetArgs.N;

% Set seeds for initial condition, define constant for model, set initial
% condition and solve KS equation
rng(200);
init_seed = randi(10000,1,16);
const = 1;%[1e-3,1e-2,1e-1,1];
rng(init_seed(6));
u = 0.6*(-1+2*rand([ModelParams.N,1]));

[~,data] = solve_kursiv(u,ModelParams);
%ModelParams.const = const(1);

% Run prediction method
[valid_time,norm_rmse] = hybrid_listen_predict_opts(NetArgs,ModelParams,data,init_seed(1),const,rmse_max,sys_type,plotif,N,winput);