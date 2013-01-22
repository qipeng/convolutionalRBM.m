function res = whiten_data(input, useCuda)
    if (nargin < 3),
        useCuda = 0;
    end
    
    w = 7;
    halfw = floor(w/2);
    whM = zeros(w);
    for i = 1:w,for j = 1:w,whM(i,j)=exp(-abs(i-halfw-1)^2)*exp(-abs(j-halfw-1)^2)*evensign(i+j-halfw*2-2);end;end
 
    [M, N, ~, n] = size(input);
    padded = zeros(M, N, n+2*halfw, n+2*halfw);
    for i = 1:M, for j = 1:N, padded(i, j, halfw + 1:end - halfw, halfw + 1:end - halfw) = input(i,j,:,:);end;end
    
    res = reshape(convs(padded, reshape(whM, [1 1 w w]), useCuda), [M N n n]);
    
%     for i = 1:M,
%         for j = 1:N,
%             for x = 1:n-w+1,
%                 for y = 1:n-w+1,
%                     res(i,j,x:x+w-1,y:y+w-1) = res(i,j,x:x+w-1,y:y+w-1)...
%                         + reshape(whM * reshape(input(i,j,x:x+w-1,y:y+w-1),[w*w,1]),[1 1 w w]);
%                 end
%             end
%             res(i,j,:,:) = reshape(reshape(res(i,j,:,:),n,n)./mask, [1 1 n n]);
%         end
%     end
end

function y = evensign(x)
    if mod(x,2)
        y = -1;
    else
        y = 1;
    end
end