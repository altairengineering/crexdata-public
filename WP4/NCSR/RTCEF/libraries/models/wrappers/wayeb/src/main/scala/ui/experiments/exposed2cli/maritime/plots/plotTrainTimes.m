setname = 'TrainingTimeClassification';
home = getenv('WAYEB_HOME');
resultsDir = strcat(home, '/results/maritime/');
%resultsFiles = {strcat(resultsDir,'portMeanClassification.csv'),strcat(resultsDir,'portHMMClassification.csv'),strcat(resultsDir,'portSDFAClassification.csv'),strcat(resultsDir,'portSPSAClassification.csv'),strcat(resultsDir,'portSPSTClassification.csv')};
resultsFiles = {strcat(resultsDir,'portSingleVesselDistance1MeanClassification.csv'),strcat(resultsDir,'portSingleVesselDistance1HMMClassification.csv'),strcat(resultsDir,'portSingleVesselDistance1SDFAClassification.csv'),strcat(resultsDir,'portSingleVesselDistance1SPSAClassification.csv'),strcat(resultsDir,'portSingleVesselDistance1SPSTClassification.csv')};
orderSets = {[-1], [-1], [0 1 2], [1 2 3], [1 2 3 4 5 6]};
totalOrdersNo = 14;
maxSpreads = {10,4,10,10,10};
prefices = {'MEAN','HMM','FMM','PSA','PST'};

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
set(gca,'XTickLabel',{'MEAN','HMM','IID','F1','F2','E1','E2','E3','T1','T2','T3','T4','T5','T6'});
set(gca,'XTickLabelRotation',45);
ylabel('Training time (sec)');
set(gcf,'Color','w');
set(gca,'FontSize',32);
legend(legends,'Location','northwest');
pdfTitle = strcat(setname, 'single.pdf')
export_fig(strcat(resultsDir,pdfTitle));