
maxSpreads = [0]; % [0 2 4 6];
thresholds = [0.0]; % [0.0 0.25 0.5 0.75];
distances = [1 2 3 4 5 6];
metrics = [{'score'},{'precision'},{'NoForecasts'}];

columns = [10 12 16];
%orders = [1 2 3 4 5];
%titlePrefix = 'SEQ8SDFA';
%resultsFile = '/home/zmithereen/data/feedzai/results/decreasing/decreasingSDFAFixed.csv';
orders = [0 1 2];
titlePrefix = 'IncSDFA';
resultsFile = '/home/zmithereen/data/feedzai/results/increasingSDFAFixed.csv';
results = csvread(resultsFile,1,0);
for i=1:size(metrics,2)
    decFixed(maxSpreads, distances, thresholds, orders, results, columns(i), metrics(i), titlePrefix)
end


columns = [12 14 18];
%orders = [1 2 3 4 5];
%titlePrefix = 'SEQ8SPSA';
%resultsFile = '/home/zmithereen/data/feedzai/results/decreasing/decreasingSPSAFixed.csv';
orders = [1 2 3];
titlePrefix = 'IncSPSA';
resultsFile = '/home/zmithereen/data/feedzai/results/increasingSPSAFixed.csv';
results = csvread(resultsFile,1,0);
for i=1:size(metrics,2)
    decFixed(maxSpreads, distances, thresholds, orders, results, columns(i), metrics(i), titlePrefix)
end