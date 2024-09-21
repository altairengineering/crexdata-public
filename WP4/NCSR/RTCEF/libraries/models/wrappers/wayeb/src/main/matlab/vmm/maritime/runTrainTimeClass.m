setname = 'TrainingTimeClass';

resultsDir = '/home/zmithereen/data/maritime/results/';
resultsFiles = {strcat(resultsDir,'portMeanClassification.csv'),strcat(resultsDir,'portHMMClassification.csv'),strcat(resultsDir,'portSDFAClassification.csv'),strcat(resultsDir,'portSPSAClassification.csv'),strcat(resultsDir,'portSPSTClassification.csv')};
orderSets = {[-1], [-1], [0 1 2], [1 2], [1 2 3 4 5 6]};
totalOrdersNo = 13;
maxSpreads = {10,4,10,10,10};
prefices = {'MEAN','HMM','FMM','VMM','PST'};
%resultsFiles = {strcat(resultsDir,'portSDFAClassification.csv'),strcat(resultsDir,'portSPSAClassification.csv')};
%orderSets = {[0 1 2 3], [1 2 3]};
%totalOrdersNo = 7;
%maxSpreads = {4,4};
%prefices = {'FMM','VMM'};

minDistances = [0.0 0.5];
maxDistances = [0.5 1.0];

data = zeros(size(minDistances,2),totalOrdersNo,4);

i = 1;
for which=[1 2 3 4 5]
    orders = orderSets{which};
    maxSpread = maxSpreads{which}
    resultsFile = resultsFiles{which};
    results = csvread(resultsFile,1,0);
    %prefix = strcat(setname,prefices{which});
    times = trainTimeClass(minDistances,maxDistances,orders,maxSpread,results);
    j = i + size(orders,2) - 1;
    data(:,i:j,:) = times;
    i = j + 1;
end

groupsNo = size(minDistances,2);
stacksPerGroup = totalOrdersNo;
labels = cell(groupsNo,1);
for g=1:groupsNo
    minDistance = minDistances(g);
    maxDistance = maxDistances(g);
    labels{g} = strcat(num2str(minDistance),'-',num2str(maxDistance));
end
legends = {'modelTime','wtTime','inTime','extraTime'};
plotBarStackGroups(data,labels,legends,'on');
grid on;
grid minor;
ylabel('Time (ms)');
xl = strcat('Distance (MEAN-HMM-F0-F1-F2-F3-V1-V2-V3-T1-T2-T3-T4-T5)');
xlabel(xl);
figureTitle = setname;
title(figureTitle);
set(gcf,'Color','w');
set(gca,'FontSize',26);
pdfTitle = strcat(setname, '.pdf')
export_fig(strcat(resultsDir,pdfTitle));


data2 = zeros(totalOrdersNo,4);
data2(:,:) = data(1,:,:) ./ 1000;
figure('units','normalized','outerposition',[0 0 1 1],'visible','on');
bar(data2,'stacked');
grid on;
grid minor;
set(gca,'XTickLabel',{'MEAN','HMM','F0','F1','F2','V1','V2','T1','T2','T3','T4','T5','T6'});
set(gca,'XTickLabelRotation',45);
ylabel('Training time (sec)');
set(gcf,'Color','w');
set(gca,'FontSize',32);
legend(legends,'Location','northwest');
pdfTitle = strcat(setname, 'single.pdf')
export_fig(strcat(resultsDir,pdfTitle));