function [ h ] = customPColor( varargin )
%CUSTOMPCOLOR Summary of this function goes here
%   Detailed explanation goes here
    if nargin==1
        mat=varargin{1};
        x=1:size(mat,2);
        y=1:size(mat,1);
        label=false;
    end
    if nargin==3
        mat=varargin{3};
        x=varargin{1};
        y=varargin{2};
        label=true;
    end
    xpos=(1:size(mat,2))+0.5;
    ypos=(1:size(mat,1))+0.5;
    
    xlabel={};
    for i=1:length(x)
        xlabel{i}=num2str(x(i));
    end
    ylabel={};
    for i=1:length(y)
        ylabel{i}=num2str(y(i));
    end
        
    
    mat=[mat,zeros(size(mat,1),1)];
    mat=[mat;zeros(1,size(mat,2))];
    %figure;
    gcf;
    h=pcolor(mat);
    ax=gca;
    if(label)
        set(ax,'XTick', xpos);
        set(ax,'XTickLabel', xlabel);
        set(ax,'YTick' ,ypos);
        set(ax,'YTickLabel' ,ylabel);
    else
        set(ax,'XTick', []);
        set(ax,'YTick' ,[]);
    end
    
end

