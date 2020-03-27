function [s_out,HESN] = apply_hybrid_network_kursiv(input,HESN)

% Solve for next state using the model in the hybrid
[~,u_next] = solve_kursiv(input,HESN.ModelParams);
u_next = u_next(end,:);

% Compute the next state of the reservoir
r      = (1-HESN.leakage)*HESN.r+HESN.leakage*tanh(HESN.A*HESN.r+...
         HESN.W_in*[input;u_next']+HESN.bias.*ones(length(HESN.r),1));
HESN.r = r;
r_partsquared = HESN.r;
r_partsquared(1:2:end) = r_partsquared(1:2:end).^2;

% Solve for the output
s_out  = HESN.W_out*[u_next';r_partsquared]+HESN.c_star;
return