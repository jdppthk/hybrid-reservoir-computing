function [s_out,HESN] = apply_reservoir_only_network_kursiv_winput(input,HESN)
% Applies reservoir network with feed-forward term (u_next)
u_next = HESN.U(:,end);
r      = (1-HESN.leakage)*HESN.r+HESN.leakage*tanh(HESN.A*HESN.r+...
         HESN.W_in*[input;u_next]+HESN.bias.*ones(length(HESN.r),1));

HESN.r = r;
r_partsquared = HESN.r;
r_partsquared(1:2:end) = r_partsquared(1:2:end).^2;

% Calculate output
s_out  = HESN.W_out*[u_next;r_partsquared;input]+HESN.c_star;
return