maxSpread = 0;
threshold = 0.0;
distances = [1 2 3 4 5];

%column = 8;
%scoreType = 'RMSE';

column = 7;
scoreType = 'ANOIS';

%column = 16;
%scoreType = 'Size';

orders = [-1];
resultsFile = '/home/zmithereen/data/feedzai/results/increasingMeanFixed.csv';
results = csvread(resultsFile,1,0);
scores = fixed(maxSpread, threshold, orders, results, column)

orders = [-1];
resultsFile = '/home/zmithereen/data/feedzai/results/increasingHMMFixed.csv';
results = csvread(resultsFile,1,0);
scores = [scores; fixed(maxSpread, threshold, orders, results, column)]

orders = [0 1 2 3];
resultsFile = '/home/zmithereen/data/feedzai/results/increasingSDFAFixed.csv';
results = csvread(resultsFile,1,0);
scores = [scores; fixed(maxSpread, threshold, orders, results, column)]

orders = [1 2 3];
resultsFile = '/home/zmithereen/data/feedzai/results/increasingSPSAFixed.csv';
results = csvread(resultsFile,1,0);
scores = [scores; fixed(maxSpread, threshold, orders, results, column)]

orders = [1 2 3 4 5];
resultsFile = '/home/zmithereen/data/feedzai/results/increasingSPSTFixed.csv';
results = csvread(resultsFile,1,0);
scores = [scores; fixed(maxSpread, threshold, orders, results, column)]

modelLabels = {'MEAN', 'HMM', 'F0','F1','F2','F3',  'V1','V2','V3', 'T1','T2','T3','T4','T5'};
resultsDir = '/home/zmithereen/data/feedzai/results/';

figure('units','normalized','outerposition',[0 0 1 1],'visible','on');
b = bar(scores');
b(1).FaceColor = 'r';%[.2 .6 .5];
b(2).FaceColor = 'm';%[.2 .6 .5];
b(3).FaceColor = [.0 .0 .9];
b(4).FaceColor = [.25 .25 .9];
b(5).FaceColor = [.5 .5 .9];
b(6).FaceColor = [.75 .75 .9];
b(7).FaceColor = [0 0.9 0];
b(8).FaceColor = [.5 0.9 0.25];
b(9).FaceColor = [.5 0.9 0.5];
grid on;
grid minor;
ylabel(scoreType);
xlabel('Distance')
legend(modelLabels,'Location','northoutside','Orientation','horizontal');
set(gcf,'Color','w');
set(gca,'FontSize',26);
set(gca,'XTickLabel',distances);
set(gca, 'YScale', 'log');
pdfTitle = strcat(scoreType, '.pdf')
export_fig(strcat(resultsDir,pdfTitle));

