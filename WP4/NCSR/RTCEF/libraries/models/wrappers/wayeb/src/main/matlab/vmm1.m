resultsFile = '/home/zmithereen/src/Wayeb/results/psa/resultsSPSAPortDistanceMATLAB.csv';
results = csvread(resultsFile);
orders2check = [1 2 3 4]; %3 4];
maxNoStates2check = [50 100 500 1000 1250];
column = 10;


%figure;
for maxNoStates=maxNoStates2check
    scores = zeros(1,size(orders2check,2));
    i = 1;
    for order=orders2check
        index = find(results(:,1)==order & results(:,5)==maxNoStates);
        scores(i) = results(index,column);
        i = i + 1;
    end
    hold on;
    plot(orders2check,scores,':*','LineWidth',5.0,'MarkerSize',10);
end
set(gca,'XTick',orders2check);
set(gca, 'YLim',[0,50000]);
xlabel('Order');
%ylabel('Training time (ms)');
ylabel('Throughput (events/sec)');
set(gca,'FontSize',32);
legend({'50', '100', '500', '1000', '1250'});
set(gcf,'Color','w');