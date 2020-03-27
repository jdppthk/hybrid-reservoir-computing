function s_out = reservoir_listen_and_predict_kursiv(NetArgs,HESN,data,n)

% Reinitialize reservoir
HESN.r = zeros(NetArgs.N,1);
s_out = zeros(NetArgs.M,NetArgs.prediction_steps-NetArgs.listening_steps);

% Allow reservoir to listen up to listening_steps
for k = 1:NetArgs.listening_steps
        [~,HESN] = apply_reservoir_only_network_kursiv(data(k+NetArgs.training_steps+(n-1)*NetArgs.prediction_steps,:)',HESN);
end

% Predict up to prediction_steps
for k = 1:(NetArgs.prediction_steps-NetArgs.listening_steps)
    if k == 1
        [s_out(:,k),HESN] = apply_reservoir_only_network_kursiv(data(k+...
            NetArgs.training_steps+(n-1)*NetArgs.prediction_steps + ...
            NetArgs.listening_steps,:)',HESN);
    else
        [s_out(:,k),HESN] = apply_reservoir_only_network_kursiv(s_out(:,k-1),HESN);
    end
end
return