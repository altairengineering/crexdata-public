resultsDir = '/home/zmithereen/data/feedzai/results/';

resultsFileSdfa = strcat(resultsDir,'increasingSDFALogLoss.csv');
resultsFileSpsa = strcat(resultsDir,'increasingSPSALogLoss.csv');
resultsSdfa = csvread(resultsFileSdfa,1,0);
resultsSpsa = csvread(resultsFileSpsa,1,0);
ordersSdfa = resultsSdfa(:,1);
lossesSdfa = resultsSdfa(:,3);
statesSdfa = resultsSdfa(:,4);

pMin = 0.0001;
alpha = 0.0;
gamma = 0.001;
r = 1.05;

indices = find(resultsSpsa(:,2)==pMin & resultsSpsa(:,3)==alpha & resultsSpsa(:,4)==gamma & resultsSpsa(:,5)==r);

ordersSpsa = resultsSpsa(indices,1);
lossesSpsa = resultsSpsa(indices,6);
statesSpsa = resultsSpsa(indices,7);

losses = [lossesSdfa; lossesSpsa];
states = [statesSdfa; statesSpsa];

orderLabels = {'F1','F2','F3','V1','V2','V3','V4','V5','V6','V7','V8'};

setname = 'IncLogLoss';

figure('units','normalized','outerposition',[0 0 1 1],'visible','off');
bar(losses);
grid on;
grid minor;
set(gca, 'YLim',[min(losses)-0.01,max(losses)+0.01]);
ylabel('Average log-loss');
xlabel('Model');
%set(gca,'XTickLabelRotation',45);
figureTitle = strcat(setname,'Score');
%title(figureTitle);
set(gca,'XTickLabel',orderLabels);
set(gcf,'Color','w');
set(gca,'FontSize',32);
pdfTitle = strcat(figureTitle, '.pdf');
export_fig(strcat(resultsDir,pdfTitle));

figure('units','normalized','outerposition',[0 0 1 1],'visible','off');
bar(states);
grid on;
grid minor;
%set(gca, 'YLim',[min(losses)-0.05,max(losses)+0.05]);
ylabel('Number of states/nodes');
xlabel('Model');
%set(gca,'XTickLabelRotation',45);
figureTitle = strcat(setname,'States');
%title(figureTitle);
set(gca,'XTickLabel',orderLabels);
set(gcf,'Color','w');
set(gca,'FontSize',32);
pdfTitle = strcat(figureTitle, '.pdf');
export_fig(strcat(resultsDir,pdfTitle));