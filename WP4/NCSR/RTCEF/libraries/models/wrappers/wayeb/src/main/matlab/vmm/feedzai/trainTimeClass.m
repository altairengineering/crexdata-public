function trainTimes = trainTimeClass(minDistances, maxDistances, orders, maxSpread, results)

    groupsNo = size(minDistances,2);
    stacksPerGroup = size(orders,2);
    data = zeros(groupsNo,stacksPerGroup,4);

    for g=1:groupsNo
        minDistance = minDistances(g);
        maxDistance = maxDistances(g);
        for s=1:stacksPerGroup
            order = orders(s);
            index = find(results(:,1)==order & results(:,2)==maxSpread & results(:,3)==minDistance & results(:,4)==maxDistance);
            extraTime = mean(results(index,19));
            modelTime = mean(results(index,20));
            wtTime = mean(results(index,21));
            predTime = mean(results(index,22));
            data(g,s,1) = modelTime;
            data(g,s,2) = wtTime;
            data(g,s,3) = predTime;
            data(g,s,4) = extraTime;
         end
    end

    trainTimes = data



%{
    legends = {'modelTime','wtTime','predTime','extraTime'};
    plotBarStackGroups(data,labels,legends,'on');
    grid on;
    grid minor;
    xl = strcat('Distance (');
    for s=1:stacksPerGroup
        order = orders(s);
        xl = strcat(xl,'-',num2str(order));
    end
    ylabel('Time (ms)');
    xl = strcat(xl,')');
    xlabel(xl);
    figureTitle = strcat(prefix,'TrainingTime');
    title(figureTitle);
    set(gcf,'Color','w');
    set(gca,'FontSize',32);
    pdfTitle = strcat(prefix,figureTitle, '.pdf')
    export_fig(strcat(resultsDir,pdfTitle));
%}

end