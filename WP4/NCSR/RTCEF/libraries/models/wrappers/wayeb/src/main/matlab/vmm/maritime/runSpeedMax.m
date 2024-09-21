
resultsFileSDFA = '/home/zmithereen/data/maritime/results/speed1.0_interval600/speedSDFASpeedsMax.csv';
resultsSDFA = csvread(resultsFileSDFA,1,0);
resultsFileSPSA = '/home/zmithereen/data/maritime/results/speed1.0_interval600/speedSPSASpeedsMax.csv';
resultsSPSA = csvread(resultsFileSPSA,1,0);


ordersSDFA = [1];
ordersSPSA = [1 2 3];

maxSpreads = [2 4 6 8 10 20];
thresholds = [0.1 0.2 0.3];

metrics = [{'Score'},{'Precision'},{'Size'},{'NoForecastsRatio'},{'TrainingTime'},{'Throughput'}];
columns = [10, 12; 12, 14; 5, 6; 16, 18; 6, 8; 8, 10];

for i=1:size(metrics,2)
    speedMax(maxSpreads, thresholds, ordersSDFA, ordersSPSA, resultsSDFA, resultsSPSA, columns(i,1), columns(i,2), metrics(i), 'allVessels')
end
