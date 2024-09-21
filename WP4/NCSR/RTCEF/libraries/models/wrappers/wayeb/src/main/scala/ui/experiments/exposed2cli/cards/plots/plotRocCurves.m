setname = 'IncClassNextKROC';
home = getenv('WAYEB_HOME');
resultsDir = strcat(home, '/results/cards/');
prefices = {'MEAN', 'HMM', 'FMM', 'PSA', 'PST'};
resultsFiles = {strcat(resultsDir,'increasingMeanClassifyNextK.csv'),strcat(resultsDir,'increasingHMMClassifyNextK.csv'),strcat(resultsDir,'increasingSDFAClassifyNextK.csv'),strcat(resultsDir,'increasingSPSAClassifyNextK.csv'),strcat(resultsDir,'increasingSPSTClassifyNextK.csv')};
orderSets = {[-1],[-1],[0 1 2 3],[1 2 3 4],[1 2 3 4 5 6 7]};
modelLabels = {'MEAN', 'HMM', 'IID', 'F1','F2','F3', 'E1','E2','E3','E4',  'T1','T2','T3','T4','T5','T6','T7'};

resultSetsNo = size(resultsFiles,2);

minDistances = [0.0 0.2 0.4 0.6];
maxDistances = [0.2 0.4 0.6 0.8];

barsno = 0;
for o=orderSets
    barsno = barsno + size(o{1,1},2);
end
allaucs = zeros(size(minDistances,2),barsno);

for d=1:size(minDistances,2)
    minDistance = minDistances(d);
    maxDistance = maxDistances(d);
    aucs = [];
    for rs=1:resultSetsNo
        resultsFile = resultsFiles{rs};
        results = csvread(resultsFile,1,0);
        orders = orderSets{rs};
        prefix = strcat(setname, prefices{rs});
        aucs = [aucs roc(minDistance,maxDistance,orders,results,prefix,resultsDir)' ];
    end
    allaucs(d,:) = aucs;

    figure('units','normalized','outerposition',[0 0 1 1],'visible','off');
    bar(aucs);
    grid on;
    grid minor;
    set(gca, 'YLim',[0,1]);
    ylabel('AUC');
    figureTitle = strcat('minDist=',num2str(minDistance),'maxDist=',num2str(maxDistance));
    title(figureTitle);
    set(gca,'XTickLabel',modelLabels);
    pdfTitle = strcat(setname, 'AUC', figureTitle, '.pdf')
    export_fig(strcat(resultsDir,pdfTitle));
end

figure('units','normalized','outerposition',[0 0 1 1],'visible','off');
b = bar(allaucs);
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
ylabel('AUC');
xlabel('Distance (%)')
set(gca, 'YLim',[0,1]);
set(gca,'YTickLabels',[0 0.2 0.4 0.6 0.8 1]);
legend(modelLabels,'Location','eastoutside','Orientation','vertical');
set(gcf,'Color','w');
set(gca,'FontSize',32);
ticklabels = cell(1,size(minDistances,2));
for d=1:size(minDistances,2)
    minDistance = minDistances(d);
    maxDistance = maxDistances(d);
    label = strcat(num2str(minDistance),'-',num2str(maxDistance));
    %label = strcat(num2str(maxDistance));
    ticklabels{d} = label;
end
set(gca,'XTickLabel',ticklabels);
pdfTitle = strcat(setname, 'AUCALL', '.pdf')
export_fig(strcat(resultsDir,pdfTitle));

figure('units','normalized','outerposition',[0 0 1 1],'visible','off');
b = bar3(allaucs); %,'stacked');
v = [0.8 1.0 0.6];
view(v);
grid on;
grid minor;
set(gcf,'Color','w');
set(gca,'FontSize',32);
set(gca,'YTickLabel',ticklabels);
set(gca,'XTickLabel',modelLabels);
set(gca, 'ZLim',[0,1]);
pdfTitle = strcat(setname, 'AUCALL3d', '.pdf')
export_fig(strcat(resultsDir,pdfTitle));
