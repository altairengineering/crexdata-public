setname = 'ThroughputClassification';
home = getenv('WAYEB_HOME');
resultsDir = strcat(home, '/results/cards/');
resultsFiles = {strcat(resultsDir,'increasingMeanClassifyNextK.csv'),strcat(resultsDir,'increasingHMMClassifyNextK.csv'),strcat(resultsDir,'increasingSDFAClassifyNextK.csv'),strcat(resultsDir,'increasingSPSAClassifyNextK.csv'),strcat(resultsDir,'increasingSPSTClassifyNextK.csv')};
orderSets = {[-1], [-1], [0 1 2 3], [1 2 3 4], [1 2 3 4 5 6 7]};
totalOrdersNo = 17;
maxSpreads = {8,4,8,8,8};
prefices = {'MEAN','HMM','FMM','PSA','PST'};

minDistances = [0.0 0.2 0.4 0.6];
maxDistances = [0.2 0.4 0.6 0.8];

data1 = zeros(size(minDistances,2),totalOrdersNo);
data2 = zeros(1,totalOrdersNo);

i = 1;
for which=[1 2 3 4 5]
    orders = orderSets{which};
    maxSpread = maxSpreads{which}
    resultsFile = resultsFiles{which};
    results = csvread(resultsFile,1,0);
    [throughputs, states] = gatherThroughputs(minDistances,maxDistances,orders,maxSpread,results);
    j = i + size(orders,2) - 1;
    data1(:,i:j) = throughputs;
    data2(1,i:j) = states(1,:);
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
figure('units','normalized','outerposition',[0 0 1 1],'visible','on');
b = bar(data1);
b(1).FaceColor = 'k';%[.2 .6 .5];
b(2).FaceColor = 'm';%[.2 .6 .5];
b(3).FaceColor = 'y';%[.0 .0 .9];

b(4).FaceColor = [.25 .25 .9];
b(5).FaceColor = [.5 .5 .9];
b(6).FaceColor = [.75 .75 .9];

b(7).FaceColor = [.9 .2 .2];
b(8).FaceColor = [.9 .4 .4];
b(9).FaceColor = [.9 .6 .6];
b(10).FaceColor = [.9 .8 .8];

b(11).FaceColor = [0 0.8 0.1];
b(12).FaceColor = [0 0.8 0.2];
b(13).FaceColor = [0 0.8 0.3];
b(14).FaceColor = [0 0.8 0.4];
b(15).FaceColor = [0 0.8 0.5];
b(16).FaceColor = [0 0.8 0.6];
b(17).FaceColor = [0 0.8 0.7];
grid on;
grid minor;
xlabel('Distance (%)');
set(gca,'XTickLabel',labels);
ylabel('Throughput (events/sec)');
legend({'MEAN','HMM','IID','F1','F2','F3','E1','E2','E3','E4','T1','T2','T3','T4','T5','T6','T7'},'Location','eastoutside','Orientation','vertical');
%figureTitle = strcat('Throughput', setname);
figureTitle = setname;
%title(figureTitle);
set(gcf,'Color','w');
set(gca,'FontSize',32);
pdfTitle = strcat(figureTitle, '.pdf')
export_fig(strcat(resultsDir,pdfTitle));

figure('units','normalized','outerposition',[0 0 1 1],'visible','on');
data3 = data2;
data3(totalOrdersNo - size(orderSets{5},2) + 1:totalOrdersNo) = data3(totalOrdersNo - size(orderSets{5},2) + 1:totalOrdersNo) ./ data2(1);
bar(data3);
grid on;
grid minor;
xlabel('Model');
set(gca,'XTick',1:totalOrdersNo);
set(gca, 'XLim',[0,totalOrdersNo+1]);
set(gca,'XTickLabel',{'MEAN','HMM','IID','F1','F2','F3','E1','E2','E3','E4','T1','T2','T3','T4','T5','T6','T7'});
set(gca,'XTickLabelRotation',45);
ylabel('States/Nodes');
figureTitle = strcat('States', setname);
%title(figureTitle);
set(gcf,'Color','w');
set(gca,'FontSize',32);
pdfTitle = strcat(figureTitle, '.pdf')
export_fig(strcat(resultsDir,pdfTitle));
