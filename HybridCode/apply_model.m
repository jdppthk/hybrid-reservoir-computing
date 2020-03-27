function [s_out,MODEL] = apply_model(input,MODEL)

% Apply the model to solve for states between listening_steps and
% prediction_steps
[~,u_next] = MODEL.model(input,MODEL.ModelParams);
s_out = u_next(end,:);

return
