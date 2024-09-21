
resultsFileSDFA= '/home/zmithereen/data/caviar/results/meetingInit/meetingInitSDFAMax.csv';
resultsSDFA = csvread(resultsFileSDFA,1,0);
resultsFileSPSA= '/home/zmithereen/data/caviar/results/meetingInit/meetingInitSPSAMax.csv';
resultsSPSA = csvread(resultsFileSPSA,1,0);


ordersSDFA= [1 2];
ordersSPSA= [1 2 3 4];

maxSpreads = [5 10 15];
thresholds = [0.25 0.5 0.75];

metrics = [{'Score'},{'Precision'},{'Size'},{'NoForecastsRatio'},{'TrainingTime'},{'Throughput'}];
columns = [10, 12; 12, 14; 5, 6; 16, 18; 6, 8; 8, 10];

for i=1:size(metrics,2)
    meetMax(maxSpreads, thresholds, ordersSDFA, ordersSPSA, resultsSDFA, resultsSPSA, columns(i,1), columns(i,2), metrics(i), 'videos10')
end

