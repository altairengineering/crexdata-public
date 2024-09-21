%resultsFile = '/home/zmithereen/src/Wayeb/resutls/psa/results.csv';
resultsFile = '/home/zmithereen/Downloads/tmp.csv';
results = csvread(resultsFile);
thres = results(:,4);
sdfa1index = 8; mergedIndex = sdfa1index+6; learntIndex = mergedIndex + 6;
sdfa1Values = results(:,sdfa1index);
mergedValues = results(:,mergedIndex);
learntValues = results(:,learntIndex);
dist = results(:,2);
spread = results(:,3);

scoreType = 'Interval Score';
experiments = 150;

for i=1:experiments
    starti = (i-1)*9+1;
    endi = starti + 8; 
    thresplot = thres(starti:endi); 
    sdfa1plot = sdfa1Values(starti:endi); 
    mergedplot = mergedValues(starti:endi);
    learntplot = learntValues(starti:endi);
    ymax = max([max(sdfa1plot),max(mergedplot),max(learntplot)]);
    thisDist = dist(starti); 
    thisSpread = spread(starti); 
    maxScore = max(sdfa1plot); 
    figure('units','normalized','outerposition',[0 0 1 1],'visible','off');
    plot(thresplot,sdfa1plot,'m--x','LineWidth',4.0); 
    hold on; plot(thresplot,learntplot,'b-.*','LineWidth',4.0);
    hold on; plot(thresplot,mergedplot,'g:d','LineWidth',4.0);
    hold on; plot(thresplot,zeros(1,9),'red','LineWidth',8.0);
    xlabel('threshold'); 
    ylabel(scoreType); 
    legend({'sdfa','learnt','merged'},'Location','southwest');
    set(gca,'YLim',[-1,ymax+1],'FontSize',32,'XTick',thresplot,'XLim',[0.1,0.9]); 
    thisTitle = strcat('distance=',num2str(thisDist),'spread=',num2str(thisSpread)); 
    title(thisTitle); 
    export_fig(strcat(thisTitle,'.pdf'))
end