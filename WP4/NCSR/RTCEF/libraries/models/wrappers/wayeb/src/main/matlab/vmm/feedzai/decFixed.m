function decFixed(maxSpreads, distances, thresholds, orders, results, column, metric, titlePrefix)

    scoreCells = cell(size(maxSpreads,2),size(thresholds,2));
    r = 0;
    for spread=maxSpreads
        r = r + 1;
        c = 0;
        for threshold=thresholds
            c = c + 1;
            scores = zeros(size(distances,2),size(orders,2));
            i = 0;
            for distance=distances
                i = i + 1;
                j = 0;
                for order=orders
                    j = j + 1;
                    index = find(results(:,1)==order & results(:,2)==spread & results(:,3)==distance & results(:,5)==threshold);
                    scores(i,j) = results(index,column);
                end
            end
            scoreCells{r,c} = scores;
        end
    end

    r = 0;
    for spread=maxSpreads
        r = r + 1;
        c = 0;
        for threshold=thresholds
            c = c + 1;
            figure('units','normalized','outerposition',[0 0 1 1],'visible','off');
            y = scoreCells{r,c};


            %b = bar3(y); %,'stacked');
            %v = [0.5 1.0 1.0];
            %view(v);
            %b = bar(y);
            %xlabel('Order');
            %ylabel('Distance');
            %zlabel(metric);
            %set(gca,'XTickLabel',orders);
            %set(gca,'YTickLabel',distances);

            b = bar(y);
            %{
            width = b.BarWidth;
            for i=1:length(y(:, 1))
                row = y(i, :);
                % 0.5 is approximate net width of white spacings per group
                offset = ((width + 0.5) / length(row)) / 2;
                x = linspace(i-offset, i+offset, length(row));
                text(x,row,num2str(row'),'vert','bottom','horiz','center');
            end
            %}
            xlabel('Distance');
            ylabel(metric);
            if (size(orders,2)==1)
                legend({'m=1'});
            elseif (size(orders,2)==2)
                legend({'m=1','m=2'});
            elseif (size(orders,2)==3)
                legend({'m=1','m=2','m=3'});
            elseif (size(orders,2)==4)
                legend({'m=1','m=2','m=3','m=4'});
            else
                legend({'m=1','m=2','m=3','m=4','m=5'});
            end
            set(gca,'XTickLabel',distances);

            set(gca,'FontSize',32);
            figureTitle = strcat('spread=',num2str(spread),'threshold=',num2str(threshold));
            title(figureTitle);
            set(gcf,'Color','w');
            pdfTitle = strcat(titlePrefix,'DecFixed',metric{1,1},'Spread',num2str(spread),'Threshold',num2str((100*threshold)),'.pdf')
            export_fig(pdfTitle);
        end
    end


end
