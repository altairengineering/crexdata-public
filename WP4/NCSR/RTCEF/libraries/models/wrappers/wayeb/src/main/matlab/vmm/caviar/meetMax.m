function meetMax(maxSpreads, thresholds, ordersSDFA, ordersSPSA, resultsSDFA, resultsSPSA, columnSDFA, columnSPSA, metric, titlePrefix)

    scoresSDFA = zeros(size(maxSpreads,2)*size(thresholds,2),size(ordersSDFA,2));
    scoresSPSA = zeros(size(maxSpreads,2)*size(thresholds,2),size(ordersSPSA,2));
    r = 0;
    maxScore = 0;
    for spread=maxSpreads
        for threshold=thresholds
            r = r + 1;

            i = 1;
            for order=ordersSDFA
                index = find(resultsSDFA(:,1)==order & resultsSDFA(:,2)==spread & resultsSDFA(:,4)==threshold);
                scoresSDFA(r,i) = resultsSDFA(index,columnSDFA);
                i = i + 1;
            end

            i = 1;
            for order=ordersSPSA
                index = find(resultsSPSA(:,1)==order & resultsSPSA(:,2)==spread & resultsSPSA(:,4)==threshold);
                scoresSPSA(r,i) = resultsSPSA(index,columnSPSA);
                i = i + 1;
            end

        end
    end

    r = 0;
    for spread=maxSpreads
        for threshold=thresholds
            r = r + 1;
            figure('units','normalized','outerposition',[0 0 1 1],'visible','off');

            currentScores = scoresSDFA(r,:);
            %validIndices = find(currentScores ~= -1);
            %plot(ordersSDFADist(validIndices),currentScores(validIndices),'--*','LineWidth',5.0,'MarkerSize',15);
            plot(ordersSDFA,currentScores,'--*','LineWidth',5.0,'MarkerSize',15);

            currentScores = scoresSPSA(r,:);
            %validIndices = find(currentScores ~= -1);
            hold on;
            %plot(ordersSPSADist(validIndices),currentScores(validIndices),'-.o','LineWidth',5.0,'MarkerSize',15);
            plot(ordersSPSA,currentScores,'--o','LineWidth',5.0,'MarkerSize',15);

            plot(ordersSPSA,zeros(1,size(ordersSPSA,2)),'red','LineWidth',8.0);

            set(gca,'XTick',ordersSPSA);
            %set(gca, 'YLim',[-1,maxScore]);
            xlabel('Order');
            ylabel(metric);
            set(gca,'FontSize',32);
            legend({'FMM', 'VMM'});
            figureTitle = strcat('spread=',num2str(spread),'threshold=',num2str(threshold));
            title(figureTitle);
            set(gcf,'Color','w');
            pdfTitle = strcat(titlePrefix, 'MeetMax',metric{1,1}, 'Spread', num2str(spread), 'Threshold', num2str(100*threshold), '.pdf')
            export_fig(pdfTitle);
        end
    end
end
