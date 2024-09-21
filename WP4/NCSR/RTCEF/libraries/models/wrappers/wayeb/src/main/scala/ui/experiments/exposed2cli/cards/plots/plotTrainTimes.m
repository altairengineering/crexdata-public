setname = 'TrainingTimeClassification';
home = getenv('WAYEB_HOME');
resultsDir = strcat(home, '/results/cards/');
resultsFiles = {strcat(resultsDir,'increasingMeanClassifyNextK.csv'),strcat(resultsDir,'increasingHMMClassifyNextK.csv'),strcat(resultsDir,'increasingSDFAClassifyNextK.csv'),strcat(resultsDir,'increasingSPSAClassifyNextK.csv'),strcat(resultsDir,'increasingSPSTClassifyNextK.csv')};
orderSets = {[-1], [-1], [0 1 2 3], [1 2 3 4], [1 2 3 4 5 6 7]};
totalOrdersNo = 17;
maxSpreads = {8,4,8,8,8};
prefices = {'MEAN','HMM','FMM','PSA','PST'};

minDistances = [0.0 0.2 0.4 0.6];
maxDistances = [0.2 0.4 0.6 0.8];

data = zeros(size(minDistances,2),totalOrdersNo,4);

i = 1;
for which=[1 2 3 4 5]
    orders = orderSets{which};
    maxSpread = maxSpreads{which}
    resultsFile = resultsFiles{which};
    results = csvread(resultsFile,1,0);
    %prefix = strcat(setname,prefices{which});
    times = gatherTrainTimes(minDistances,maxDistances,orders,maxSpread,results);
    j = i + size(orders,2) - 1;
    data(:,i:j,:) = times;
    i = j + 1;
end

legends = {'modelTime','wtTime','inTime','extraTime'};
data2 = zeros(totalOrdersNo,4);
data2(:,:) = data(1,:,:) ./ 1000;
figure('units','normalized','outerposition',[0 0 1 1],'visible','on');
bar(data2,'stacked');
grid on;
grid minor;
set(gca,'XTick',1:totalOrdersNo);
set(gca, 'XLim',[0,totalOrdersNo+1]);
set(gca,'XTickLabel',{'MEAN','HMM','IID','F1','F2','F3','E1','E2','E3','E4','T1','T2','T3','T4','T5','T6','T7'});
set(gca,'XTickLabelRotation',45);
ylabel('Training time (sec)');
xlabel('Model');
set(gcf,'Color','w');
set(gca,'FontSize',32);
legend(legends,'Location','northwest');
pdfTitle = strcat(setname, 'single.pdf')
export_fig(strcat(resultsDir,pdfTitle));