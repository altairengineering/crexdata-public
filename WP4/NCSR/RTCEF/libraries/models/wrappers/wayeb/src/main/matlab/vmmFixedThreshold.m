resultsFile = '/home/zmithereen/src/Wayeb/results/psa/results.csv';
%resultsFile = '/home/zmithereen/Downloads/tmp.csv';
results = csvread(resultsFile);
thres = results(:,4);
sdfa1index = 8; 
%mergedIndex = sdfa1index + 6; 
learntIndex = sdfa1index + 9;
sdfa1Values = results(:,sdfa1index);
%mergedValues = results(:,mergedIndex);
learntValues = results(:,learntIndex);
ymax = max(sdfa1Values);
dist = results(:,2);
spread = results(:,3);

scoreType = 'score';
tests = 5;
experiments = 1;
xpoints = 2;
linesPerTest = experiments * xpoints;

for i=1:experiments
    starti = (i-1)*xpoints+1;
    endi = starti + xpoints - 1; 
    distplot = dist(starti:endi); 
    xmin = min(distplot);
    xmax = max(distplot);
    sdfaall = zeros(tests,xpoints);
    learntall = zeros(tests,xpoints);
    %sdfa1plot = sdfa1Values(starti:endi); 
    %mergedplot = mergedValues(starti:endi);
    %learntplot = learntValues(starti:endi);
    %ymax = max([max(sdfa1plot),max(mergedplot),max(learntplot)]);
    for i=1:tests
        thisTestStart = (i-1)*linesPerTest + starti;
        thisTestEnd = (i-1)*linesPerTest + endi;
        sdfaall(i,:) = sdfa1Values(thisTestStart:thisTestEnd);
        learntall(i,:) = learntValues(thisTestStart:thisTestEnd);
    end
    sdfa1plot = sum(sdfaall)./tests;
    learntplot = sum(learntall)./tests;
    ymax = max([max(sdfa1plot),max(learntplot)]);
    thisSpread = spread(starti); 
    maxScore = max(sdfa1plot); 
    figure('units','normalized','outerposition',[0 0 1 1],'visible','on');
    plot(distplot,sdfa1plot,'m--x','LineWidth',4.0,'MarkerSize',10); 
    hold on; plot(distplot,learntplot,'b-.*','LineWidth',4.0,'MarkerSize',10);
    %hold on; plot(distplot,mergedplot,'g:d','LineWidth',4.0,'MarkerSize',10);
    %hold on; plot(spreadplot,spreadplot,'g','LineWidth',8.0);
    hold on; plot(distplot,zeros(1,xpoints),'red','LineWidth',8.0);
    xlabel('distance'); 
    ylabel(scoreType); 
    %legend({'sdfa','learnt','merged'},'Location','southeast');
    legend({'sdfa','learnt'},'Location','southeast');
    set(gca,'YLim',[-1,ymax+1],'FontSize',32,'XTick',distplot,'XLim',[xmin,xmax]);%,'YLim',[0,ymax]); 
    thisTitle = strcat('spread=',num2str(thisSpread)); 
    title(thisTitle); 
    %export_fig(strcat(thisTitle,'.pdf'))
end