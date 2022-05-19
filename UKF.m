%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x_n,P_n]=UKF(f,x_n1,P,h,y_n,Q,R)
L=numel(x_n1);
alpha=1;
ki=3-L;
beta=2; % for gaussian
lambda=alpha^2*(L+ki)-L;
Wm=[lambda/(L+lambda) 1/(2*(lambda+L))*ones(1,2*L)];
Wc=Wm;
Wc(1)=lambda/(L+lambda)+(1-alpha^2+beta);

Y = repmat(x_n1, 1,L);
Xsigmaset = [x_n1 Y+sqrt(L+lambda)*chol(P)' Y-sqrt(L+lambda)*chol(P)'];

LL=2*L+1;
x_np=zeros(L,1);
Xsigmap=zeros(L,LL);
y_np=zeros(L,1);
y_xsigmap=zeros(L,LL);
for i=1:LL
    Xsigmap(:,i)=f(Xsigmaset(:,i));
    y_xsigmap(:,i)=h(Xsigmap(:,i));
    x_np=x_np+Wm(i)*Xsigmap(:,i);
    y_np=y_np+Wm(i)*y_xsigmap(:,i);
end

P_np=(Xsigmap-x_np(:,ones(1,LL)))*diag(Wc)*(Xsigmap-x_np(:,ones(1,LL)))'+Q;

Pyy=(y_xsigmap-y_np(:,ones(1,LL)))*diag(Wc)*(y_xsigmap-y_np(:,ones(1,LL)))'+R;
Pxy=(Xsigmap-x_np(:,ones(1,LL)))*diag(Wc)*(y_xsigmap-y_np(:,ones(1,LL)))';
K=Pxy/Pyy;
x_n=x_np+K*(y_n-y_np);
P_n=P_np-K*Pxy';
end
