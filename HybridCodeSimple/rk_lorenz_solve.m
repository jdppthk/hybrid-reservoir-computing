function soln = rk_lorenz_solve(init_cond,ModelParams)

%initialize structures
t=zeros(ModelParams.nstep+1,1);
t(1) = 0;
y = zeros(3,ModelParams.nstep+1);
y(:,1) = init_cond;

for i=1:ModelParams.nstep
    %update time
    t(i+1)=t(i)+ModelParams.tau;
    %update vector
    %y is handle of solution vector
    y(:,i+1) = rk4(@f, y(:,i), t(i), ModelParams);
end

soln = y';
return
