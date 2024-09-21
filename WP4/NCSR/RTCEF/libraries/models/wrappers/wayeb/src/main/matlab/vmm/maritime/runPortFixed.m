
maxSpreads = [5 10 15];
thresholds = [0.25 0.5 0.75];
distances = [60 180 300 420 540];

metrics = [{'score'},{'precision'},{'NoForecasts'}];







columns = [10 12 16];

%{
maxSpreads = [5 10 15];
orders = [1 2];
titlePrefix = 'allVesselsDistSDFA';
resultsFile = '/home/zmithereen/data/maritime/results/port1.0/portSDFADistancesFixed.csv';
%}


maxSpreads = [2 4 6 8 10 20];
orders = [1];
distances = [600 1200 1800 2400 3000 3600];
thresholds = [0.1 0.2 0.3];
titlePrefix = 'allVesselsSpeedsSDFA';
resultsFile = '/home/zmithereen/data/maritime/results/speed1.0_interval600/speedSDFASpeedsFixed.csv';

results = csvread(resultsFile,1,0);

for i=1:size(metrics,2)
    portFixed(maxSpreads, distances, thresholds, orders, results, columns(i), metrics(i), titlePrefix)
end


%{
orders = [1 2];
resultsFile = '/home/zmithereen/data/maritime/results/port1.0/portSDFADistancesHeadingFixed.csv';
results = csvread(resultsFile,1,0);
titlePrefix = 'allVesselsDistHeadSDFA';
for i=1:size(metrics,2)
    portFixed(maxSpreads, distances, thresholds, orders, results, columns(i), metrics(i), titlePrefix)
end
%}

columns = [12 14 18];
%{
orders = [1 2 3 4];
titlePrefix = 'allVesselsDistSPSA';
resultsFile = '/home/zmithereen/data/maritime/results/port1.0/portSPSADistancesFixed.csv';
%}
%{
maxSpreads = [2 4 6 8 10 20];
orders = [1 2 3];
distances = [600 1200 1800 2400 3000 3600];
thresholds = [0.1 0.2 0.3];
titlePrefix = 'allVesselsSpeedsSPSA';
resultsFile = '/home/zmithereen/data/maritime/results/speed1.0_interval600/speedSPSASpeedsFixed.csv';
results = csvread(resultsFile,1,0);

for i=1:size(metrics,2)
    portFixed(maxSpreads, distances, thresholds, orders, results, columns(i), metrics(i), titlePrefix)
end
%}
%{
orders = [1 2 3];
resultsFile = '/home/zmithereen/data/maritime/results/port1.0/portSPSADistancesHeadingFixed.csv';
results = csvread(resultsFile,1,0);
titlePrefix = 'allVesselsDistHeadSPSA';
for i=1:size(metrics,2)
    portFixed(maxSpreads, distances, thresholds, orders, results, columns(i), metrics(i), titlePrefix)
end
%}







%{
columns = [10 12 16];
orders = [1 2];
resultsFile = '/home/zmithereen/data/maritime/results/port1.0/portSDFADistancesFixed227592820.csv';
results = csvread(resultsFile,1,0);
titlePrefix = '227592820DistSDFA';
for i=1:size(metrics,2)
    portFixed(maxSpreads, distances, thresholds, orders, results, columns(i), metrics(i), titlePrefix)
end
%}

%{
orders = [1 2];
resultsFile = '/home/zmithereen/data/maritime/results/port1.0/portSDFADistancesHeadingFixed227592820.csv';
results = csvread(resultsFile,1,0);
titlePrefix = '227592820DistHeadSDFA';
for i=1:size(metrics,2)
    portFixed(maxSpreads, distances, thresholds, orders, results, columns(i), metrics(i), titlePrefix)
end
%}
%{

columns = [12 14 18];
orders = [1 2 3 4];
resultsFile = '/home/zmithereen/data/maritime/results/port1.0/portSPSADistancesFixed227592820.csv';
results = csvread(resultsFile,1,0);
titlePrefix = '227592820DistSPSA';
for i=1:size(metrics,2)
    portFixed(maxSpreads, distances, thresholds, orders, results, columns(i), metrics(i), titlePrefix)
end
%}

%{
orders = [1 2 3];
resultsFile = '/home/zmithereen/data/maritime/results/port1.0/portSPSADistancesHeadingFixed227592820.csv';
results = csvread(resultsFile,1,0);
titlePrefix = '227592820DistHeadSPSA';
for i=1:size(metrics,2)
    portFixed(maxSpreads, distances, thresholds, orders, results, columns(i), metrics(i), titlePrefix)
end
%}