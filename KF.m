function [x_n,K_n] = KF (x_n1,K_n1,y_n,f,fdf,Q_n,C_n,R_n)
F_n = diag(fdf(x_n1));

K_nn1 = F_n*K_n1*F_n' + Q_n;
x_nn1 = f(x_n1);

G = K_nn1*C_n'/(C_n*K_nn1*C_n' + R_n);
alpha = y_n-C_n*x_nn1;
x_n = x_nn1 + G*alpha;
K_n = K_nn1 - G*C_n*K_nn1;
end