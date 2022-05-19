clear all;
close all;
clc;


delt = 0.01;
a12=0.0025;
a21=0.005;
v = 1;
u = 40;
Q_n = v^2*eye(2);
R = [u^2 0;0 u^2];

H_n=[1 0;0 1]; % Measurement Matrix
x(1:2,1) = [400; 100];

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
    % State propagation
    v_n = v*randn(2,1);
    x(1:2,n) = f(x(1:2,n-1)) + v_n;


    % Generate measurements
    w_n = u*randn(2,1);
    y_n = H_n*x(1:2,n) + w_n;
    
    %Linearized KF
    [xkf(1:2,n),Pkf(1:2,1:2,n)] = KF (xkf(1:2,n-1),Pkf(1:2,1:2,n-1),y_n, f,fdf,Q_n,H_n,R);
    
    %EKF
    [xekf(1:2,n),P(1:2,1:2,n)] = EKF(xekf(1:2,n-1),P(1:2,1:2,n-1),y_n, f,fJf,Q_n,H_n,R);

    %UKF
    [xukf(:,n), Pukf] = UKF(f,xukf(:,n-1),Pukf,h,y_n,Q_n,R);

    
end

t = [0:1999]'*delt;
figure
hold on
plot(t, x(1,:), 'b-');
plot(t, x(2,:), 'b-');

plot(t, xkf(1,:), 'r-');
plot(t, xkf(2,:), 'r-');
legend('Real x(1)','real x(2)','LKF estimate x(1)', 'LKF estimate x(2)');
xlabel('Time')
ylabel('population')
title('Tracking with LKF')
hold off

figure
hold on
plot(t, x(1,:), 'b-');
plot(t, x(2,:), 'b-');

plot(t, xekf(1,:), 'r-');
plot(t, xekf(2,:), 'r-');
legend('Real x(1)','real x(2)','EKF estimate x(1)', 'EKF estimate x(2)');
xlabel('Time')
ylabel('population')
title('Tracking with EKF')
hold off

figure
hold on
plot(t, x(1,:), 'b-');
plot(t, x(2,:), 'b-');

plot(t, xukf(1,:), 'r-');
plot(t, xukf(2,:), 'r-');
legend('Real x(1)','real x(2)','UKF estimate x(1)', 'UKF estimate x(2)');
xlabel('Time')
ylabel('population')
title('Tracking with UKF')
hold off

