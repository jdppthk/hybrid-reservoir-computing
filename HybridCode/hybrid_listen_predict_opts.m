function [valid_time,norm_rmse,s_out] = hybrid_listen_predict_opts(NetArgs,ModelParams,data,seed,const,rmse_max,sys_type,plotif,N,winput)
% close all

NetArgs.N = N;
NetArgs.bias  = zeros(NetArgs.N,1);

%% Like hybrid_predict_lorenz if sys_type is 'hybrid', shuts off reservoir if sys_type is 'model', shuts off model if sys_type is 'reservoir
% if system type not hybrid, model, or reservoir 
if ~strcmpi(sys_type,'hybrid') && ~strcmpi(sys_type,'model') && ~strcmpi(sys_type,'reservoir')
    error('Enter hybrid, model, or reservoir')
end

rng(seed)
%% Generate and train the hybrid Echo state network
if strcmpi(sys_type,'hybrid')
    HESN_base = generate_HESN(NetArgs);
end

if strcmpi(sys_type,'reservoir')
    ESN_base = generate_partzeroed_HESN(NetArgs);
else
    NewModelParams = ModelParams;
    NewModelParams.nstep = 1;
    NewModelParams.const = const;
end

% Define prediction matrix
s_out = zeros(NetArgs.M,NetArgs.prediction_steps-NetArgs.listening_steps,NetArgs.num_predictions);

% If only model, don't worry about beta to get s_out
% HESN --> MODEL
if strcmpi(sys_type,'model')
    MODEL = train_model(data,data,ModelParams.model,NewModelParams,NetArgs);
%     trained_s = MODEL.trained_s;
    for n = 1:NetArgs.num_predictions
        s_out(:,:,n) = model_listen_and_predict(NetArgs,MODEL,data,n);        
    end
end

% For non-model methods, train the network for either method and use it to
% predict
if ~strcmpi(sys_type,'model')
    beta = 1e-5;
    if strcmpi(sys_type,'hybrid')
        if winput == 1
            HESN = train_hybrid_network_kursiv_winput(data,data,NewModelParams,beta,HESN_base,NetArgs);
        else
            HESN = train_hybrid_network_kursiv(data,data,NewModelParams,beta,HESN_base,NetArgs);
        end
    end
    if strcmpi(sys_type,'reservoir')
        if winput == 1
            HESN = train_reservoir_only_network_kursiv_winput(data,data,beta,ESN_base,NetArgs);
        else
            HESN = train_reservoir_only_network_kursiv(data,data,beta,ESN_base,NetArgs);
        end
    end

    %% Inputting the state of the Kurumoto-Sivashinsky equation, use the 
    % hybrid system to solve for the state of the system
    if strcmpi(sys_type,'hybrid')
        if winput == 1
            for n = 1:NetArgs.num_predictions
                s_out(:,:,n) = hybrid_listen_and_predict_kursiv_winput(NetArgs,HESN,data,n);
            end
        else
            for n = 1:NetArgs.num_predictions
                s_out(:,:,n) = hybrid_listen_and_predict_kursiv(NetArgs,HESN,data,n);
            end
        end
    end
    % Use only the reservoir to solve
    if strcmpi(sys_type,'reservoir')
        if winput == 1
            for n = 1:NetArgs.num_predictions
                s_out(:,:,n) = reservoir_listen_and_predict_kursiv_winput(NetArgs,HESN,data,n);
            end
        else
            for n = 1:NetArgs.num_predictions
                s_out(:,:,n) = reservoir_listen_and_predict_kursiv(NetArgs,HESN,data,n);
            end
        end
    end

    if plotif == 1
        % Plot weights of W_out
        figure
        %distribution of weights on nodes (the first M are weights of model direct connection)
        var_to_plot = 1; %x is 1, y is 2, z is 3
        plot(HESN.W_out(var_to_plot,:),'+')
        title('Wout Weights on model output direct connection + on each node')
        xlabel('model output + nodes')
        ylabel('Wout weights')
        hold on
        x = 1:NetArgs.M+NetArgs.N;
        y = zeros(1,NetArgs.M+NetArgs.N);
        plot(x,y)
    end
end


%% RMSE method for calculating valid_length
%normalized rmse
valid_time = zeros(1,NetArgs.num_predictions);
norm_rmse  = zeros(NetArgs.prediction_steps-NetArgs.listening_steps,NetArgs.num_predictions);
rmse_max2 = 0.8;
for n = 1:NetArgs.num_predictions
    
    err = data((NetArgs.training_steps+2+NetArgs.listening_steps+(n-1)*NetArgs.prediction_steps):...
            (NetArgs.training_steps+1+n*NetArgs.prediction_steps),:) - s_out(:,:,n)';
%    norm_rmse(:,n) = sqrt(sum(err.^2,2)./sum(data((NetArgs.training_steps+2+NetArgs.listening_steps+(n-1)*NetArgs.prediction_steps):...
%            (NetArgs.training_steps+1+n*NetArgs.prediction_steps),:).^2,2));
    norm_rmse(:,n) = sqrt(sum(err.^2,2))./sqrt(mean(sum(data((NetArgs.training_steps+2+NetArgs.listening_steps+(n-1)*NetArgs.prediction_steps):...
        (NetArgs.training_steps+1+n*NetArgs.prediction_steps),:).^2,2)));
% Find when norm_rmse passes limit
    valid_length = 1;
    valid_length2 = 1;
    % while all parameters rmse(t)<1 and valid_length is less than end of data
    while norm_rmse(valid_length,n) < rmse_max && valid_length < size(s_out,2)
        valid_length = valid_length + 1;
    end
    
    % Give a warning if the valid length extends to the end of the data
    if valid_length == size(s_out,2)
         warning('Valid length extends longer than length of data, consider increasing solution time.')
    end
    
    % Valid_length ends before rmse exceeds max
    valid_length = valid_length - 1;   

    valid_time(n) = valid_length*ModelParams.tau;
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    
    while norm_rmse(valid_length2,n) < rmse_max2 && valid_length < size(s_out,2)
        valid_length2 = valid_length2 + 1;
    end
    
    % Give a warning if the valid length extends to the end of the data
    if valid_length2 == size(s_out,2)
         warning('Valid length extends longer than length of data, consider increasing solution time.')
    end
    
    % Valid_length ends before rmse exceeds max
    valid_length2 = valid_length2 - 1;
   

    valid_time2 = valid_length2*ModelParams.tau;
    
    
end
%% Plot if you want to (data after and before training)
%after training

xl = [0,14];
if plotif == 1
    % Plotting spatiotemporal models
    figure
    subplot(3,1,1)
    colormap('jet')
    x = 1:ModelParams.N;
    t = (1:ModelParams.nstep - NetArgs.training_steps -NetArgs.listening_steps)*ModelParams.tau*0.07;%((1:ModelParams.nstep - NetArgs.training_steps)*ModelParams.tau).*0.07;
    imagesc(t,x,data(NetArgs.training_steps+NetArgs.listening_steps + 2:end,:)');
    title([sys_type ' prediction'])
    xlabel('time')
    ylabel('Expected')
    xlim(xl);
    colorbar;
    subplot(3,1,2)
    colormap('jet')
    x = 1:ModelParams.N;
    t = (1:ModelParams.nstep - NetArgs.training_steps -NetArgs.listening_steps)*ModelParams.tau*0.07;%((1:ModelParams.nstep - NetArgs.training_steps)*ModelParams.tau).*0.07;
    prediction_plot = zeros(ModelParams.N,length(t));
    for n=1:NetArgs.num_predictions
        prediction_plot(:,(NetArgs.listening_steps+1+(n-1)*NetArgs.prediction_steps):(NetArgs.prediction_steps*n))=...
            s_out(:,:,n);
    end
    imagesc(t,x,prediction_plot(x,(NetArgs.listening_steps+1+(n-1)*NetArgs.prediction_steps):(NetArgs.prediction_steps*n)));
    xlabel('time')
    ylabel('Predicted')
    colorbar;
    xlim(xl)
    subplot(3,1,3)
    x = 1:ModelParams.N;
    t = (1:ModelParams.nstep - NetArgs.training_steps -NetArgs.listening_steps)*ModelParams.tau*0.07;%((1:ModelParams.nstep - NetArgs.training_steps)*ModelParams.tau).*0.07;
    diff = zeros(ModelParams.N,length(t));
    for n = 1:NetArgs.num_predictions
        diff(:,(NetArgs.listening_steps+1+(n-1)*NetArgs.prediction_steps):(NetArgs.prediction_steps*n)) = ...
            data((NetArgs.training_steps+NetArgs.listening_steps+(n-1)*NetArgs.prediction_steps+2):...
            (NetArgs.training_steps+1+n*NetArgs.prediction_steps),:)'-s_out(:,:,n);
    end
    colormap('jet')
    imagesc(t,x,diff(x,(NetArgs.listening_steps+1+(n-1)*NetArgs.prediction_steps):(NetArgs.prediction_steps*n)));
    line([valid_time*0.07, valid_time*0.07], [0,64])
    line([valid_time2*0.07, valid_time2*0.07], [0,64])
    xlabel('time')
    ylabel('Difference')
    xlim(xl)
    colorbar;
end
%%

return