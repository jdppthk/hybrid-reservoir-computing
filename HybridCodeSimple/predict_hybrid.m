function output = predict_hybrid(y,x,resparams,ModelParams,A,win,wout)


output = zeros(resparams.num_inputs, resparams.predict_length);
out = wout*(vertcat(y,x));
output(:,1) = out;
for i = 1:resparams.predict_length
    x = tanh(A*x + win*out);
    y = lorenz_step_forward(out, ModelParams);%rk4(@f, out, 0, ModelParams);
    out = wout*(vertcat(y',x));
    output(:,i+1) = out;
end