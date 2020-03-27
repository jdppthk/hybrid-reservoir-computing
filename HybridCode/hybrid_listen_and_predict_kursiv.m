function s_out = hybrid_listen_and_predict_kursiv(NetArgs,HESN,data,n)

% Reinitialize reservoir
HESN.r = zeros(NetArgs.N,1);
s_out = zeros(NetArgs.M,NetArgs.prediction_steps-NetArgs.listening_steps);

% Listen to input for listening steps
for k = 1:NetArgs.listening_steps
        [~,HESN] = apply_hybrid_network_kursiv(data(k+NetArgs.training_steps+(n-1)*NetArgs.prediction_steps,:)',HESN);
end

% Predict after this using hybrid up to prediction steps
for k = 1:(NetArgs.prediction_steps-NetArgs.listening_steps)
    if k == 1
        [s_out(:,k),HESN] = apply_hybrid_network_kursiv(data(k+NetArgs.training_steps+...
            (n-1)*NetArgs.prediction_steps+NetArgs.listening_steps,:)',HESN);
    else
        [s_out(:,k),HESN] = apply_hybrid_network_kursiv(s_out(:,k-1),HESN);
    end
end
return