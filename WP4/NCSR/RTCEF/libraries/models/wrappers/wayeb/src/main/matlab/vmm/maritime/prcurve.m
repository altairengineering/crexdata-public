function aucs = roc(minDistance, maxDistance, orders, results, prefix, resultsDir)

    %thresholds = [0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0];
    thresholds = [0.0 0.1 0.3 0.5 0.7 0.9 1.0];

    scores = cell(size(orders,2),3);

    o = 0;
    for order=orders
        o = o + 1;
        recalls = zeros(size(thresholds,2),1);
        precisions = zeros(size(thresholds,2),1);
        i = 0;
        for threshold=thresholds
            i = i + 1;
            index = find(results(:,1)==order & results(:,3)==minDistance & results(:,4)==maxDistance & results(:,5)==threshold & results(:,7)==0);
            recalls(i) = results(index,9);
            precisions(i) = results(index,8);
        end
        scores{o,1} = recalls;
        scores{o,2} = precisions;
        scores{o,3} = order;
    end

    %scores

    figure('units','normalized','outerposition',[0 0 1 1],'visible','off');
    o = 0;
    legends = cell(size(orders,2),1);
    aucs = zeros(size(orders,2),1);
    for order=orders
        o = o + 1;
        recalls = scores{o,1};
        precisions = scores{o,2};
        validIndices = find(precisions(:,1)~=-1);
        recalls = [1; recalls(validIndices); 0];
        precisions = [0; precisions(validIndices); 1];
        legends{o} = num2str(order);
        aucs(o) = -trapz(recalls,precisions);
        plot(recalls,precisions,'--*','LineWidth',5.0,'MarkerSize',20);
        hold on;
    end
    grid on;
    grid minor;
    xlabel('Recall');
    ylabel('Precision');
    set(gca, 'XLim',[0,1]);
    set(gca, 'YLim',[0,1]);
    legend(legends,'Location','southwest');
    figureTitle = strcat('minDist=',num2str(minDistance),'maxDist=',num2str(maxDistance));
    title(figureTitle);
    set(gcf,'Color','w');
    set(gca,'FontSize',32);
    pdfTitle = strcat(prefix,figureTitle, '.pdf')
    export_fig(strcat(resultsDir,pdfTitle));
end