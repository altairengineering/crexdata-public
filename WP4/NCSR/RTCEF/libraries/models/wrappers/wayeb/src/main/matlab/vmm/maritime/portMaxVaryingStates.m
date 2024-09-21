function portMaxVaryingStates(results, column, metric, titlePrefix)

scores50 = zeros(4,1);
scores100 = zeros(4,1);
scores500 = zeros(4,1);
scores1000 = zeros(4,1);
spread = 5;
threshold = 0.5;
orders = [1 2 3 4];

i = 1;
for order=orders
    index = find(results(:,1)==order & results(:,2)==spread & results(:,4)==threshold & results(:,5)==50);
    scores50(i) = results(index,column);
    i = i + 1;
end

i = 1;
for order=orders
    index = find(results(:,1)==order & results(:,2)==spread & results(:,4)==threshold & results(:,5)==100);
    scores100(i) = results(index,column);
    i = i + 1;
end

i = 1;
for order=orders
    index = find(results(:,1)==order & results(:,2)==spread & results(:,4)==threshold & results(:,5)==500);
    scores500(i) = results(index,column);
    i = i + 1;
end

i = 1;
for order=orders
    index = find(results(:,1)==order & results(:,2)==spread & results(:,4)==threshold & results(:,5)==1000);
    scores1000(i) = results(index,column);
    i = i + 1;
end


figure('units','normalized','outerposition',[0 0 1 1],'visible','off');

plot(orders,scores50,'--*','LineWidth',5.0,'MarkerSize',15);
hold on;
plot(orders,scores100,'--x','LineWidth',5.0,'MarkerSize',15);
hold on;
plot(orders,scores500,'--o','LineWidth',5.0,'MarkerSize',15);
hold on;
plot(orders,scores1000,'--^','LineWidth',5.0,'MarkerSize',15);
plot(orders,zeros(1,size(orders,2)),'red','LineWidth',8.0);

set(gca,'XTick',orders);
xlabel('Order');
ylabel(metric);
set(gca,'FontSize',32);
legend({'maxNoStates=50', 'maxNoStates=100', 'maxNoStates=500', 'maxNoStates=1000'});
figureTitle = strcat('spread=',num2str(spread),'threshold=',num2str(threshold));
title(figureTitle);
set(gcf,'Color','w');
pdfTitle = strcat(titlePrefix, 'PortMaxVaryingStates',metric{1,1}, 'Spread', num2str(spread), 'Threshold', num2str(100*threshold), '.pdf')
export_fig(pdfTitle);

end