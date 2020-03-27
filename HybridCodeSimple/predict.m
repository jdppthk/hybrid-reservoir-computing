function [output,x] = predict(A,win,resparams,x, w_out)

out = w_out*x;

output = zeros(resparams.num_inputs, resparams.predict_length);

for i = 1:resparams.predict_length
    x = tanh(A*x + win*out);
    out = w_out*x;
    output(:,i) = out;
end
