fn = '/home/elias/ownCloud/data/debs17/wt_example/wt.csv';

results = csvread(fn);

%thresholds = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9];
target_state = 3;
thresholds = [0.5];
%cmap = hsv(100);
for ti=1:1
    thres = thresholds(ti);
    [rti, cti] = find(results==thres);
    %rti
    h = figure('Visible','on','Position', [100, 100, 1600, 800]);
    %legend('-DynamicLegend'); 
    hold all;
    for r=rti'
        %r
        state = results(r,1);
        l = results(r,4);
        l20 = 12;
        times = 1:l20;
        probs = results(r,5:l20+5-1);
        istart = results(r,l+5);
        iend = results(r,l+6);
        ci = randi([1,100]);
        if (state==0) 
            color = 'red';
            marker = '*';
        elseif (state==1)
            color = 'green';
            marker = 'o';
        elseif (state==2)
            color = 'blue';
            marker = 's';
        else
            color = 'magenta';
            marker = 'd';
        end
        h1 = plot(times,probs,'Marker',marker,'Color',color,'LineWidth',5.0,'MarkerSize',10.0,'DisplayName',strcat('state:',num2str(state)));
        if (state ~= target_state)
            h1.Color(4)=0.2;
        end
        if (state==target_state)
            h2 = plot([istart, iend],[1.0, 1.0],'--x','Color',color,'LineWidth',5.0,'MarkerSize',30.0,'DisplayName',strcat('interval:',num2str(istart),',',num2str(iend)));
        end
        
        %h2.Color(4)=0.3;
        axis([gca],[1 l20 0.0 1.0]);
        set(gca,'Xtick',linspace(times(1),times(l20),l20),'LineWidth',5.0);
        %plot([istart,iend],[-1,-1],'DisplayName',strcat('interval',num2str(state)));
        legend('-DynamicLegend');
        %hold all;
    end
    %plot([1, l20],[thres, thres],'--','Color','black','LineWidth',2.0,'DisplayName',strcat('threshold:',num2str(thres)));
    %title(gca,'Prediction Accuracy per state');
    set(gca,'FontSize',40);
    set(gcf, 'Color', 'w');
    xlabel(gca,'Number of future events');
    ylabel(gca,'Completion Probability');
    legend(gca,'show');
    
    legend(gca,'show');
    %figname = strcat('/home/elias/ownCloud/data/debs17/wt_example/wt',num2str(target_state),'.png');
    %print(figname, '-depsc', '-r300')
    %export_fig -test.png % -q101
    figname = strcat('/home/elias/ownCloud/data/debs17/wt_example/wt',num2str(target_state));
    export_fig(figname, '-pdf');
    %saveas(gcf,strcat('/home/elias/ownCloud/data/debs17/wt_example/wt',num2str(target_state),'_',num2str(thres),'.png'),'png')
end