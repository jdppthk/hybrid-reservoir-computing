function s_out = hybrid_listen_and_predict_kursiv_winput(NetArgs,HESN,data,n)

% Reinitialize the reservoir
HESN.r = zeros(NetArgs.N,1);
s_out = zeros(NetArgs.M,NetArgs.prediction_steps-NetArgs.listening_steps);

% Listen to the input for the set number of listening steps
for k = 1:NetArgs.listening_steps
        [~,HESN] = apply_hybrid_network_kursiv_winput(data(k+NetArgs.training_steps+(n-1)*NetArgs.prediction_steps,:)',HESN);
end

% Predict with the hybrid for the set number of prediction steps
for k = 1:(NetArgs.prediction_steps-NetArgs.listening_steps)
    if k == 1
        [s_out(:,k),HESN] = apply_hybrid_network_kursiv_winput(data(k+NetArgs.training_steps+...
            (n-1)*NetArgs.prediction_steps+NetArgs.listening_steps,:)',HESN);
    else
        [s_out(:,k),HESN] = apply_hybrid_network_kursiv_winput(s_out(:,k-1),HESN);
    end
end
return