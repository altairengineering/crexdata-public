function portMax(maxSpreads, thresholds, ordersSDFADist, ordersSDFADistHead, ordersSPSADist, ordersSPSADistHead, resultsSDFADist, resultsSDFADistHead, resultsSPSADist, resultsSPSADistHead, columnSDFADist, columnSDFADistHead, columnSPSADist, columnSPSADistHead, metric, titlePrefix)

    scoresSDFADist = zeros(size(maxSpreads,2)*size(thresholds,2),size(ordersSDFADist,2));
    scoresSDFADistHead = zeros(size(maxSpreads,2)*size(thresholds,2),size(ordersSDFADistHead,2));
    scoresSPSADist = zeros(size(maxSpreads,2)*size(thresholds,2),size(ordersSPSADist,2));
    scoresSPSADistHead = zeros(size(maxSpreads,2)*size(thresholds,2),size(ordersSPSADistHead,2));
    r = 0;
    maxScore = 0;
    for spread=maxSpreads
        for threshold=thresholds
            r = r + 1;

            i = 1;
            for order=ordersSDFADist
                index = find(resultsSDFADist(:,1)==order & resultsSDFADist(:,2)==spread & resultsSDFADist(:,4)==threshold);
                scoresSDFADist(r,i) = resultsSDFADist(index,columnSDFADist);
                i = i + 1;
            end

            i = 1;
            for order=ordersSDFADistHead
                index = find(resultsSDFADistHead(:,1)==order & resultsSDFADistHead(:,2)==spread & resultsSDFADistHead(:,4)==threshold);
                scoresSDFADistHead(r,i) = resultsSDFADistHead(index,columnSDFADistHead);
                i = i + 1;
            end

            i = 1;
            for order=ordersSPSADist
                index = find(resultsSPSADist(:,1)==order & resultsSPSADist(:,2)==spread & resultsSPSADist(:,4)==threshold);
                scoresSPSADist(r,i) = resultsSPSADist(index,columnSPSADist);
                i = i + 1;
            end

            i = 1;
            for order=ordersSPSADistHead
                index = find(resultsSPSADistHead(:,1)==order & resultsSPSADistHead(:,2)==spread & resultsSPSADistHead(:,4)==threshold);
                scoresSPSADistHead(r,i) = resultsSPSADistHead(index,columnSPSADistHead);
                i = i + 1;
            end
        end
    end

    r = 0;
    for spread=maxSpreads
        for threshold=thresholds
            r = r + 1;
            figure('units','normalized','outerposition',[0 0 1 1],'visible','off');

            currentScores = scoresSDFADist(r,:);
            %validIndices = find(currentScores ~= -1);
            %plot(ordersSDFADist(validIndices),currentScores(validIndices),'--*','LineWidth',5.0,'MarkerSize',15);
            plot(ordersSDFADist,currentScores,'--*','LineWidth',5.0,'MarkerSize',15);

            currentScores = scoresSDFADistHead(r,:);
            %validIndices = find(currentScores ~= -1);
            hold on;
            %plot(ordersSDFADistHead(validIndices),currentScores(validIndices),'--x','LineWidth',5.0,'MarkerSize',15);
            plot(ordersSDFADistHead,currentScores,'--x','LineWidth',5.0,'MarkerSize',15);

            currentScores = scoresSPSADist(r,:);
            %validIndices = find(currentScores ~= -1);
            hold on;
            %plot(ordersSPSADist(validIndices),currentScores(validIndices),'-.o','LineWidth',5.0,'MarkerSize',15);
            plot(ordersSPSADist,currentScores,'--o','LineWidth',5.0,'MarkerSize',15);

            currentScores = scoresSPSADistHead(r,:);
            %validIndices = find(currentScores ~= -1);
            hold on;
            %plot(ordersSPSADistHead(validIndices),currentScores(validIndices),'-^','LineWidth',5.0,'MarkerSize',15);
            plot(ordersSPSADistHead,currentScores,'--^','LineWidth',5.0,'MarkerSize',15);

            plot(ordersSPSADist,zeros(1,size(ordersSPSADist,2)),'red','LineWidth',8.0);

            set(gca,'XTick',ordersSPSADist);
            %set(gca, 'YLim',[-1,maxScore]);
            xlabel('Order');
            ylabel(metric);
            set(gca,'FontSize',32);
            legend({'FMM (distance)', 'FMM (distance + heading)', 'VMM (distance)', 'VMM (distance + heading)'});
            figureTitle = strcat('spread=',num2str(spread),'threshold=',num2str(threshold));
            title(figureTitle);
            set(gcf,'Color','w');
            pdfTitle = strcat(titlePrefix, 'PortMax',metric{1,1}, 'Spread', num2str(spread), 'Threshold', num2str(100*threshold), '.pdf')
            export_fig(pdfTitle);
        end
    end
end
