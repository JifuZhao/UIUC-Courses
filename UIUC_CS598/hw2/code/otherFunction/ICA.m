% ICA method
function Z=ICA(X)

[M, T] = size(X);
average= mean(X')';  %均值

for i=1:M     
    X(i,:)=X(i,:)-average(i)*ones(1,T);  
end 

%%-------Whiten-------------------
Cx = cov(X',1);    %计算协方差矩阵Cx  
[eigvector,eigvalue] = eig(Cx); %计算Cx的特征值和特征向量 
W=eigvalue^(-1/2)*eigvector';   %白化矩阵 
% [U, S, V] = svd(cov(X'));
% W=S^(-1/2)*U';   %whiten matrix 
Z=W*X;   %正交矩阵 

%----------迭代-------  
Maxcount=10000;        %最大迭代次数 
Critical=0.000001;   %判断是否收敛  
m=M;                %需要估计的分量的个数 
W=rand(m);  
for n=1:m        
    WP=W(:,n);  %初始权矢量（任意） 
    %Y=WP'*Z;  
    %     G=Y.^3;
    %G为非线性函数，可取y^3等 
    %     GG=3*Y.^2;  %G的导数
    count=0;      
    LastWP=zeros(m,1);      
    W(:,n)=W(:,n)/norm(W(:,n));      
    while abs(WP-LastWP)&abs(WP+LastWP)>Critical         
        count=count+1;   %迭代次数         
        LastWP=WP;      %上次迭代的值         
        % WP=1/T*Z*((LastWP'*Z).^3)'-3*LastWP;         
        for i=1:m       
            WP(i)=mean(Z(i,:).*(tanh((LastWP)'*Z)))-(mean(1-(tanh((LastWP))'*Z).^2)).*LastWP(i);
        end
        WPP=zeros(m,1);         
        for j=1:n-1             
            WPP=WPP+(WP'*W(:,j))*W(:,j);        
        end
        WP=WP-WPP;         
        WP=WP/(norm(WP));                 
        if count==Maxcount             
            fprintf('未找到相应的信号');       
            return;          
        end
    end
    W(:,n)=WP; 
end

Z=W'*Z;