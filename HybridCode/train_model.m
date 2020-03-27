function MODEL = train_model(input,output,model,ModelParams,NetArgs)
% This function doesn't do much but retains the structure of methods with a
% reservoir to be trained

%% Set number of training steps and check if data is large enough
if NetArgs.training_steps+1 > size(input,1)
    error('Not enough data to train on.')
end


%% Calculate best matrix for network states and expected output

% Output should be perfect model at all training steps, so output a "trained
% s"
MODEL.trained_s = output(2:NetArgs.training_steps+1,:)';
MODEL.model = model;
MODEL.ModelParams = ModelParams;

return