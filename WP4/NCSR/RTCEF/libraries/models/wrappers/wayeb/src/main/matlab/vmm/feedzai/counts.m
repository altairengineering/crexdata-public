function aucs = counts(threshold, orders, results, prefix, resultsDir)

    minDistances = [0.0 0.2 0.4 0.6 0.8];
    maxDistances = [0.2 0.4 0.6 0.8 1.0];

    groupsNo = size(minDistances,2);
    stacksPerGroup = size(orders,2);
    data = zeros(groupsNo,stacksPerGroup,4);
    labels = cell(groupsNo,1);

    for g=1:groupsNo
        minDistance = minDistances(g);
        maxDistance = maxDistances(g);
        labels{g} = strcat(num2str(minDistance),'-',num2str(maxDistance));
        for s=1:stacksPerGroup
            order = orders(s);
            index = find(results(:,1)==order & results(:,3)==minDistance & results(:,4)==maxDistance & results(:,5)==threshold);
            tp = results(index,12);
            tn = results(index,13);
            fp = results(index,14);
            fn = results(index,15);
            data(g,s,1) = tp;
            data(g,s,2) = tn;
            data(g,s,3) = fp;
            data(g,s,4) = fn;
        end
     end

    %figure('units','normalized','outerposition',[0 0 1 1],'visible','off');
    legends = {'tp','tn','fp','fn'};
    plotBarStackGroups(data,labels,legends,'off');
    grid on;
    grid minor;
    xl = strcat('Distance (', prefix);
    for s=1:stacksPerGroup
        order = orders(s);
        xl = strcat(xl,'-',num2str(order));
    end
    ylabel('Counts');
    xl = strcat(xl,')');
    xlabel(xl);
    %legend({'0','1','2'})
    figureTitle = strcat('threshold=',num2str(threshold));
    title(figureTitle);
    set(gcf,'Color','w');
    set(gca,'FontSize',32);
    pdfTitle = strcat(prefix,figureTitle, '.pdf')
    export_fig(strcat(resultsDir,pdfTitle));
end