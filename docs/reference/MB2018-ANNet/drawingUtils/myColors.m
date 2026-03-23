function [ colors ] = myColors( n )
%MYCOLORS Summary of this function goes here
%   Detailed explanation goes here
if(n==1)
    colors=[0,0,0];
else
    colors=zeros(n,3);
    for i=1:n
        slot=floor((i-1)*6/n);
        x=mod((i-1)*6/n,1);
        switch slot
            case 0
                colors(i,:)=[1   x   0];
            case 1
                colors(i,:)=[1-x 1   0];
            case 2
                colors(i,:)=[0   1   x];
            case 3
                colors(i,:)=[0   1-x 1];
            case 4
                colors(i,:)=[x   0   1];
            case 5
                colors(i,:)=[1   0   1-x];
        end
    end
end
end

