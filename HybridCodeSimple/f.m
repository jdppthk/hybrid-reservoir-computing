function df = f(t,x,ModelParams)




df = zeros(3,1);
df(1) = -1*ModelParams.a*x(1) + ModelParams.a*x(2);
df(2) = ModelParams.b*x(1) - x(2) - x(1)*x(3);
df(3) = -ModelParams.c*x(3) + x(1)*x(2);
end