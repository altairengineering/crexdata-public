resultsFile = '/home/elias/src/Wayeb/results/psa/results.csv';
%resultsFile = '/home/zmithereen/Downloads/tmp.csv';
results = csvread(resultsFile);

sdfa = results(6:9);
spsa = results(12:15);

P = [sdfa; spsa];

P_labels = {'Precision';'Spread';'Score';'Recall'};

% Figure properties
 figure('units', 'normalized', 'outerposition', [0 0.05 1 0.95]);

% Axes properties
 axes_interval = 5;
 axes_precision = 1;

% Spider plot
 spider_plot(P, P_labels, axes_interval, axes_precision,...
 'Marker', 'o',...
 'LineStyle', '-',...
 'LineWidth', 2,...
 'MarkerSize', 5);

% Title properties
 title('Sample Spider Plot',...
 'Fontweight', 'bold',...
 'FontSize', 12);

% Legend properties
 legend('show', 'Location', 'southoutside');