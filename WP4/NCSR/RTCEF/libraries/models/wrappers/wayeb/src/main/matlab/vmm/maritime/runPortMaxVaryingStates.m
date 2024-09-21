resultsFile = '/home/zmithereen/data/maritime/results/port1.0/portSPSADistancesMaxVaryingStates.csv';
results = csvread(resultsFile,1,0);
metrics = [{'Score'},{'Precision'},{'Size'},{'NoForecastsRatio'},{'TrainingTime'},{'Throughput'}];
columns = [12; 14; 6; 18; 8; 10];

for i=1:size(metrics,2)
    portMaxVaryingStates(results, columns(i), metrics(i), 'allVessels')
end