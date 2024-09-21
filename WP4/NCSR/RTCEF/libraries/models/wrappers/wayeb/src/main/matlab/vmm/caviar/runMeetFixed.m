
maxSpreads = [5 10 15];
thresholds = [0.25 0.5 0.75];
distances = [1 2 3 4 5 6 7 8 9 10];
metrics = [{'score'},{'precision'},{'NoForecasts'}];

columns = [10 12 16];
orders = [1 2];
titlePrefix = 'videos10SDFA';
resultsFile = '/home/zmithereen/data/caviar/results/meetingInit/meetingInitSDFAFixed.csv';
results = csvread(resultsFile,1,0);
for i=1:size(metrics,2)
    meetFixed(maxSpreads, distances, thresholds, orders, results, columns(i), metrics(i), titlePrefix)
end


columns = [12 14 18];
orders = [1 2 3 4];
titlePrefix = 'videos10SPSA';
resultsFile = '/home/zmithereen/data/caviar/results/meetingInit/meetingInitSPSAFixed.csv';
results = csvread(resultsFile,1,0);
for i=1:size(metrics,2)
    meetFixed(maxSpreads, distances, thresholds, orders, results, columns(i), metrics(i), titlePrefix)
end