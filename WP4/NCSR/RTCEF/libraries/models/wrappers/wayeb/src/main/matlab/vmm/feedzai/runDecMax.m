
%resultsFileSDFA= '/home/zmithereen/data/feedzai/results/decreasing/decreasingSDFAMax.csv';
%resultsFileSPSA= '/home/zmithereen/data/feedzai/results/decreasing/decreasingSPSAMax.csv';
resultsFileSDFA= '/home/zmithereen/data/feedzai/results/increasingSDFAMax.csv';
resultsFileSPSA= '/home/zmithereen/data/feedzai/results/increasingSPSAMax.csv';
resultsSDFA = csvread(resultsFileSDFA,1,0);
resultsSPSA = csvread(resultsFileSPSA,1,0);


%ordersSDFA= [1 2 3 4 5];
%ordersSPSA= [1 2 3 4 5];
ordersSDFA= [1 2];
ordersSPSA= [1 2 3 4];

maxSpreads = [0 2 4 6];
thresholds = [0.0 0.25 0.5 0.75];

metrics = [{'Score'},{'Precision'},{'Size'},{'NoForecastsRatio'},{'TrainingTime'},{'Throughput'}];
columns = [10, 12; 12, 14; 5, 6; 16, 18; 6, 8; 8, 10];

for i=1:size(metrics,2)
    decMax(maxSpreads, thresholds, ordersSDFA, ordersSPSA, resultsSDFA, resultsSPSA, columns(i,1), columns(i,2), metrics(i), 'Paths')
end

