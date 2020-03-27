function s_out = model_listen_and_predict(NetArgs,MODEL,data,n)
s_out = zeros(NetArgs.M,NetArgs.prediction_steps-NetArgs.listening_steps);

% Predict using the model up to prediction_steps
for k = 1:(NetArgs.prediction_steps-NetArgs.listening_steps)
    if k==1
        [s_out(:,k),MODEL] = apply_model(data(k+NetArgs.training_steps+...
            (n-1)*NetArgs.prediction_steps+NetArgs.listening_steps,:)',MODEL);
    else
        [s_out(:,k),MODEL] = apply_model(s_out(:,k-1),MODEL);
    end
end

return