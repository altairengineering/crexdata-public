setname = 'Class';

resultsDir = '/home/zmithereen/data/maritime/results/';
resultsFiles = {strcat(resultsDir,'portMeanClassification.csv'),strcat(resultsDir,'portHMMClassification.csv'),strcat(resultsDir,'portSDFAClassification.csv'),strcat(resultsDir,'portSPSAClassification.csv'),strcat(resultsDir,'portSPSTClassification.csv')};
orderSets = {[-1], [-1], [0 1 2], [1 2], [1 2 3 4 5 6]};
totalOrdersNo = 13;
maxSpreads = {10,4,10,10,10};
prefices = {'MEAN','HMM','FMM','VMM','PST'};

minDistances = [0.0 0.5];
maxDistances = [0.5 1.0];

data1 = zeros(size(minDistances,2),totalOrdersNo);
data2 = zeros(1,totalOrdersNo);

i = 1;
for which=[1 2 3 4 5]
    orders = orderSets{which};
    maxSpread = maxSpreads{which}
    resultsFile = resultsFiles{which};
    results = csvread(resultsFile,1,0);
    [throughputs, states] = throughputClass(minDistances,maxDistances,orders,maxSpread,results);
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
b(1).FaceColor = 'r';%[.2 .6 .5];
b(2).FaceColor = 'm';%[.2 .6 .5];
b(3).FaceColor = [.0 .0 .9];
b(4).FaceColor = [.25 .25 .9];
b(5).FaceColor = [.5 .5 .9];
%b(6).FaceColor = [.75 .75 .9];
b(6).FaceColor = [0 0.9 0];
b(7).FaceColor = [.5 0.9 0.25];
%b(8).FaceColor = [.5 0.9 0.5];
grid on;
grid minor;
xlabel('Distance (%)');
set(gca,'XTickLabel',labels);
ylabel('Throughput (events/sec)');
legend({'MEAN','HMM','F0','F1','F2','V1','V2','T1','T2','T3','T4','T5','T6'},'Location','northoutside','Orientation','horizontal');
figureTitle = strcat('Throughput', setname);
%title(figureTitle);
set(gcf,'Color','w');
set(gca,'FontSize',26);
pdfTitle = strcat(figureTitle, '.pdf')
export_fig(strcat(resultsDir,pdfTitle));

figure('units','normalized','outerposition',[0 0 1 1],'visible','on');
bar(data2);
grid on;
grid minor;
xlabel('Model');
set(gca,'XTickLabel',{'MEAN','HMM','F0','F1','F2','V1','V2','T1','T2','T3','T4','T5','T6'});
set(gca,'XTickLabelRotation',45);
ylabel('States');
%legend({'MEAN','HMM','FMM0','FMM1','FMM2','VMM1','VMM2'})
figureTitle = strcat('States', setname);
%title(figureTitle);
set(gcf,'Color','w');
set(gca,'FontSize',32);
pdfTitle = strcat(figureTitle, '.pdf')
export_fig(strcat(resultsDir,pdfTitle));
