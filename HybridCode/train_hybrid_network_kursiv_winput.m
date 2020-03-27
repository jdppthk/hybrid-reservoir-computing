function HESN = train_hybrid_network_kursiv_winput(input,output,ModelParams,beta,HESN,NetArgs)
%% Set number of training steps and check if data is large enough
% training_steps = 260;
if NetArgs.training_steps+1 > size(input,1)
    error('Not enough data to train on.')
end

%% Calculate the results of the hybrid system with feedforward input for 
% this set of data
r = zeros(length(HESN.r),NetArgs.training_steps+1);
r(:,1) = HESN.r;
U = zeros(NetArgs.M,NetArgs.training_steps);
for k=(1:NetArgs.training_steps)+1
    [~,u_next] = solve_kursiv(input(k-1,:)',ModelParams);
    u_next = u_next(end,:);
    U(:,k-1) = u_next';
    r(:,k) = (1-NetArgs.leakage)*r(:,k-1)+NetArgs.leakage*tanh(HESN.A*r(:,k-1)+HESN.W_in*[input(k-1,:)';u_next']+NetArgs.bias.*ones(NetArgs.N,1));
end
r = r(:,2:end);
HESN.r      = r(:,end);
r(1:2:end,:) = r(1:2:end,:).^2;

%% Calculate best matrix for network states and expected output

S = output(1:NetArgs.training_steps+1,:)*(1-NetArgs.noise/2)+...
    NetArgs.noise*rand(size(output(1:NetArgs.training_steps+1,:),1),...
    size(output(1:NetArgs.training_steps+1,:),2)).*...
    output(1:NetArgs.training_steps+1,:);

W_out = (S(2+NetArgs.transient_length:end,:)'*...
    [U(:,1+NetArgs.transient_length:end);r(:,1+NetArgs.transient_length:end);...
    input(1+NetArgs.transient_length:NetArgs.training_steps,:)']')*...
    pinv([U(:,1+NetArgs.transient_length:end);r(:,1+NetArgs.transient_length:end);...
    input(1+NetArgs.transient_length:NetArgs.training_steps,:)']*...
    [U(:,1+NetArgs.transient_length:end);r(:,1+NetArgs.transient_length:end);...
    input(1+NetArgs.transient_length:NetArgs.training_steps,:)']' + beta*speye(NetArgs.N+2*NetArgs.M));

% c_star = -(W_out*r_bar-s_bar);

HESN.full_r = r;
HESN.W_out  = W_out;
HESN.c_star = zeros(size(output,2),1);
HESN.U      = U;
HESN.ModelParams = ModelParams;

return