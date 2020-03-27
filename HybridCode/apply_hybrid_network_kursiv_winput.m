function [s_out,HESN] = apply_hybrid_network_kursiv_winput(input,HESN)

% Solve for next state using model in hybrid
[~,u_next] = solve_kursiv(input,HESN.ModelParams);
u_next = u_next(end,:);

% Find the next reseroir state
r      = (1-HESN.leakage)*HESN.r+HESN.leakage*tanh(HESN.A*HESN.r+...
         HESN.W_in*[input;u_next']+HESN.bias.*ones(length(HESN.r),1));
HESN.r = r;
r_partsquared = HESN.r;
r_partsquared(1:2:end) = r_partsquared(1:2:end).^2;

% Calculate output
s_out  = HESN.W_out*[u_next';r_partsquared;input]+HESN.c_star;
return