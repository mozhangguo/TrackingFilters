clear all;
close all;
clc;

%State space model 


ensumbled_lc = zeros(6,2000);
processing_t = zeros(3,100);
for trial=1:100
d = zeros(6,2000);

delt = 0.01;
a12=0.0025;
a21=0.005;
v = 1;
u = 40;
Q_n = v^2*eye(2);
R = [u^2 0;0 u^2];

H_n=[1 0;0 1]; % Measurement Matrix
x(1:2,1) = [400; 100];
P(1:2,1:2,1) = eye(2);

% EKF init
xekf(1:2,1) = x(1:2,1);
P(1:2,1:2,1) = eye(2);

% KF init
xkf(1:2,1) = [400; 100]; % Initial channel coefficients
Pkf(1:2,1:2,1) = eye(2); % Gaussian posterior covariance at time step 1


% UKF init
xukf(1:2,1) = [400; 100];
Pukf = eye(2);

f = @(x)[(1+delt*(1-x(2,1)*a21)) * x(1,1);(1-delt*(1-x(1,1)*a12)) * x(2,1)];
fdf = @(x)[1+delt*(1-x(2)*a21);1-delt*(1-x(1)*a12)];
fJf = @(x)[1+delt*(1-x(2)*a21) -delt*x(1)*a21;delt*x(2)*a12 1-delt*(1-x(1)*a12)];
h = @(x)[x(1);x(2)];
    for n=2:2000
        v_n = v*randn(2,1);
        x(1:2,n) = f(x(1:2,n-1)) + v_n;
    
    
        w_n = u*randn(2,1);
        y_n = H_n*x(1:2,n) + w_n;
        
         %Linearized KF
        tic    
        [xkf(1:2,n),Pkf(1:2,1:2,n)] = KF (xkf(1:2,n-1),Pkf(1:2,1:2,n-1),y_n, f,fdf,Q_n,H_n,R);
        processing_t(1,trial) = toc + processing_t(1,trial);        
        d(1:2,n) = abs(x(1:2,n) - xkf(1:2,n));

        tic
        [xekf(1:2,n),P(1:2,1:2,n)] = EKF(xekf(1:2,n-1),P(1:2,1:2,n-1),y_n, f,fJf,Q_n,H_n,R);
        processing_t(2,trial) = toc + processing_t(2,trial);
        d(3:4,n) = abs(x(1:2,n) - xekf(1:2,n));
        

        % Compute for the ukf
        tic
        [xukf(:,n), Pukf] = UKF(f,xukf(:,n-1),Pukf,h,y_n,Q_n,R);
        processing_t(3,trial) = toc + processing_t(3,trial);
        d(5:6,n) = abs(x(1:2,n) - xukf(1:2,n));

    end
    ensumbled_lc = ensumbled_lc + d.^2;
end

ensumbled_lc = ensumbled_lc/100;
t = [0:1999]'*delt;
figure
plot(t, ensumbled_lc(1,:))
hold on, plot(t, ensumbled_lc(3,:))
plot(t, ensumbled_lc(5,:))
legend('KF','EKF','UKF')
xlabel('Time')
ylabel('RMSE')
title('RMSE performance of x(1) tracking')
hold off

figure
plot(t, ensumbled_lc(2,:))
hold on, plot(t, ensumbled_lc(4,:))
plot(t, ensumbled_lc(6,:))
legend('KF','EKF','UKF')
xlabel('Time')
ylabel('RMSE')
title('RMSE performance of x(2) tracking')
hold off

figure
hold on
plot(1:100, processing_t(1,1:100), 'ro');
plot(1:100, processing_t(2,1:100) , 'b^');
plot(1:100, processing_t(3,1:100)),'cd';
legend('KF','EKF','UKF')
title('processing time')

