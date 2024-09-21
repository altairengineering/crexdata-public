fnTotal = '/home/zmithereen/src/Wayeb/results/datacron/maritime/approachingBrestPort/singleVesselExtrasTotalm1.csv';
resultsTotal = csvread(fnTotal);
thresholds = resultsTotal(:,2);
counts = resultsTotal(:,3);
precisions = resultsTotal(:,4);
spreads = resultsTotal(:,5);
distances = resultsTotal(:,6);

figure;
bar(thresholds,counts);
xlabel('Confidence threshold');
ylabel('Total number of predictions');
set(gca,'XTick',thresholds,'YLim',[0,15000]);
title('Number of predictions from all states');
export_fig('approachingExtrasSingleVesselm1CountsTotal.pdf')

figure;
bar(thresholds,precisions);
xlabel('Confidence threshold');
ylabel('Precision');
set(gca,'XTick',thresholds,'YLim',[0,1.0]);
title('Precision from all states');
export_fig('approachingExtrasSingleVesselm1PrecisionsTotal.pdf')

figure;
bar(thresholds,spreads);
xlabel('Confidence threshold');
ylabel('Spread');
set(gca,'XTick',thresholds,'YLim',[0,100]);
title('Spread from all states');
export_fig('approachingExtrasSingleVesselm1SpreadsTotal.pdf')

figure;
bar(thresholds,distances);
xlabel('Confidence threshold');
ylabel('Distance');
set(gca,'XTick',thresholds,'YLim',[0,50]);
title('Distance from all states');
export_fig('approachingExtrasSingleVesselm1DistancesTotal.pdf')

fnByState = '/home/zmithereen/src/Wayeb/results/datacron/maritime/approachingBrestPort/singleVesselExtrasByStatem1.csv';
resultsByState = csvread(fnByState);
thresholds = [0.1 0.3 0.5 0.7 0.9];
states = [0 1 2 3 4];

countsByState = zeros(size(thresholds,2),size(states,2));
for ti=1:size(thresholds,2)
    threshold = thresholds(ti);
    countsi = resultsByState(find(resultsByState(:,2)==threshold),4);
    countsByState(ti,:) = countsi;
end
figure;
bar(thresholds,countsByState);
xlabel('Confidence threshold');
ylabel('Number of predictions');
set(gca,'XTick',thresholds,'YLim',[0,15000]);
title('Number of predictions per state');
legend({'state=0','state=1','state=2','state=3','state=4'},'Location','northwest');
export_fig('approachingExtrasSingleVesselm1CountsByState.pdf')

precisionsByState = zeros(size(thresholds,2),size(states,2));
for ti=1:size(thresholds,2)
    threshold = thresholds(ti);
    precisionsi = resultsByState(find(resultsByState(:,2)==threshold),5);
    precisionsByState(ti,:) = precisionsi;
end
figure;
bar(thresholds,precisionsByState);
xlabel('Confidence threshold');
ylabel('Precision');
set(gca,'XTick',thresholds,'YLim',[0,1.0]);
title('Precision per state');
legend({'state=0','state=1','state=2','state=3','state=4'},'Location','northwest');
export_fig('approachingExtrasSingleVesselm1PrecisionsByState.pdf')


spreadsByState = zeros(size(thresholds,2),size(states,2));
for ti=1:size(thresholds,2)
    threshold = thresholds(ti);
    spreadsi = resultsByState(find(resultsByState(:,2)==threshold),6);
    spreadsByState(ti,:) = spreadsi;
end
figure;
bar(thresholds,spreadsByState);
xlabel('Confidence threshold');
ylabel('Spread');
set(gca,'XTick',thresholds,'YLim',[0,100]);
title('Spread per state');
legend({'state=0','state=1','state=2','state=3','state=4'},'Location','northwest');
export_fig('approachingExtrasSingleVesselm1SpreadsByState.pdf')

distancesByState = zeros(size(thresholds,2),size(states,2));
for ti=1:size(thresholds,2)
    threshold = thresholds(ti);
    distancesi = resultsByState(find(resultsByState(:,2)==threshold),7);
    distancesByState(ti,:) = distancesi;
end
figure;
bar(thresholds,distancesByState);
xlabel('Confidence threshold');
ylabel('Distance');
set(gca,'XTick',thresholds,'YLim',[0,50]);
title('Distance per state');
legend({'state=0','state=1','state=2','state=3','state=4'},'Location','northwest');
export_fig('approachingExtrasSingleVesselm1DistancesByState.pdf')







