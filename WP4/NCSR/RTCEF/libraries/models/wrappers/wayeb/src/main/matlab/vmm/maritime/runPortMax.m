
resultsFileSDFADist = '/home/zmithereen/data/maritime/results/port1.0/portSDFADistancesMax.csv';
resultsSDFADist = csvread(resultsFileSDFADist,1,0);
resultsFileSDFADistHead = '/home/zmithereen/data/maritime/results/port1.0/portSDFADistancesHeadingMax.csv';
resultsSDFADistHead = csvread(resultsFileSDFADistHead,1,0);
resultsFileSPSADist = '/home/zmithereen/data/maritime/results/port1.0/portSPSADistancesMax.csv';
resultsSPSADist = csvread(resultsFileSPSADist,1,0);
resultsFileSPSADistHead = '/home/zmithereen/data/maritime/results/port1.0/portSPSADistancesHeadingMax.csv';
resultsSPSADistHead = csvread(resultsFileSPSADistHead,1,0);


ordersSDFADist = [1 2];
ordersSDFADistHead = [1 2];
ordersSPSADist = [1 2 3 4];
ordersSPSADistHead = [1 2 3];

maxSpreads = [5 10 15];
thresholds = [0.25 0.5 0.75];

metrics = [{'Score'},{'Precision'},{'Size'},{'NoForecastsRatio'},{'TrainingTime'},{'Throughput'}];
columns = [10, 10, 12, 12; 12, 12, 14, 14; 5, 5, 6, 6; 16, 16, 18, 18; 6, 6, 8, 8; 8, 8, 10, 10];

for i=1:size(metrics,2)
    portMax(maxSpreads, thresholds, ordersSDFADist, ordersSDFADistHead, ordersSPSADist, ordersSPSADistHead, resultsSDFADist, resultsSDFADistHead, resultsSPSADist, resultsSPSADistHead, columns(i,1), columns(i,2), columns(i,3), columns(i,4), metrics(i), 'allVessels')
end

%{
resultsFileSDFADist = '/home/zmithereen/data/maritime/results/port1.0/portSDFADistancesMax227592820.csv';
resultsSDFADist = csvread(resultsFileSDFADist,1,0);
resultsFileSDFADistHead = '/home/zmithereen/data/maritime/results/port1.0/portSDFADistancesHeadingMax227592820.csv';
resultsSDFADistHead = csvread(resultsFileSDFADistHead,1,0);
resultsFileSPSADist = '/home/zmithereen/data/maritime/results/port1.0/portSPSADistancesMax227592820.csv';
resultsSPSADist = csvread(resultsFileSPSADist,1,0);
resultsFileSPSADistHead = '/home/zmithereen/data/maritime/results/port1.0/portSPSADistancesHeadingMax227592820.csv';
resultsSPSADistHead = csvread(resultsFileSPSADistHead,1,0);
for i=1:size(metrics,2)
    portMax(maxSpreads, thresholds, ordersSDFADist, ordersSDFADistHead, ordersSPSADist, ordersSPSADistHead, resultsSDFADist, resultsSDFADistHead, resultsSPSADist, resultsSPSADistHead, columns(i,1), columns(i,2), columns(i,3), columns(i,4), metrics(i), '227592820')
end
%}