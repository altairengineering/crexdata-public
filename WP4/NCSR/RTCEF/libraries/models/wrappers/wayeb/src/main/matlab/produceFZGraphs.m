%resultsDir = '/mnt/Warehouse/wayeb/feedzai/real_19_01_2017/wayeb.results.2.8g';
%resultsDir = '/mnt/Warehouse/wayeb/aminess/dump/dump';
%resultsDir = '/home/zmithereen/data/fz_real_results_jan17/wayeb/wayeb.results.3.8g';
%resultsDir = '/home/zmithereen/src/Wayeb/results/dump/validation';
%resultsDir = '/home/zmithereen/data/aminess';
resultsDir = '/home/zmithereen/ownCloud/data/debs17/feedzai';

%pattern = 'ttgptt';
%pattern = 'tgpt';
pattern = 'tiiiiiii';
%pattern = 'tddddddd';
%pattern = 'ta';
%pattern = 'tfffffff';
%pname  = 'FlashAttack';
%pname  = 'FlashAttackExtra';
pname = 'IncreasingAmounts';
%pname = 'DecreasingAmounts';
%pname = 'DecreasingAmountsExtra';
%pname = 'IncreasingAmountsExtra';
%pname = 'tgpt';
%pname = 'ttgptt';
policy = 'NONOVERLAP';
order = 1;
predictionThresholds = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9];
%maxSpreads = [10 20 30 40 50];
maxSpreads = [10];

% row =  predictionThreshold(0)  maxSpread(1)            streamSize(2) execTime(3)               matchesNo(4)
%        predictionsNo(5)        correctPredictionsNo(6) predAc(7)     correctPredictionsNoGT(8) predAcGT(9)
%        allDetected(10)         missed(11)              lossRatio(12) allDetectedGT(13)
%        avgSpread(14)           avgDist(15)

%pattern = 'NorthSouth';
%pname = 'NorthSouth';
%maxSpreads = [360];
%vid = '240672000';

%pattern = 'a_aorbstar_c';
%pname = 'a_aorbstar_c';
%maxSpreads = [200];

fnid = strcat(pname,'_seq_',pattern,'_pol_',policy,'_ord_',num2str(order));
%fnid = strcat('vid_',vid,'_name_',pname,'_seq_',pattern,'_pol_',policy,'_ord_',num2str(order));

csvFileName = strcat(resultsDir,'/',fnid,'.csv');
results = csvread(csvFileName);

pti = 1;
msi = 2;
ssi = 3;
eti = 4;
mni = 5;
pni = 6;
cpni = 7;
paci = 8;
cpngti = 9;
pacgti = 10;
alldi = 11;
alldtgti = 14;
asi = 15;
adi = 16;

thresholdsNo = size(predictionThresholds,2);


for maxSpread=maxSpreads
    predAc = zeros(1,thresholdsNo);
    predAcGT = zeros(1,thresholdsNo);
    recPrecision = zeros(1,thresholdsNo);
    avgSpread = zeros(1,thresholdsNo);
    avgDist = zeros(1,thresholdsNo);
    throughput = zeros(1,thresholdsNo);
    for ti=1:thresholdsNo
        threshold = predictionThresholds(ti);
        i = find(results(:,1)==threshold & results(:,2)==maxSpread);
        predAc(ti) = results(i,paci);
        predAcGT(ti) = results(i,pacgti);
        recPrecision(ti) = results(i,alldtgti)/results(i,alldi);
        avgSpread(ti) = results(i,asi);
        avgDist(ti) = results(i,adi);
        streamSize = results(i,ssi);
        execTime = results(i,eti);
        throughput(ti) = (streamSize / execTime) * 1000000000;
    end
    hp = figure('Visible','on','Position', [100, 100, 1400, 1100]);
    hold all;
    plot(predictionThresholds,predAc,'b--o','LineWidth',5.0,'MarkerSize',10.0);
    %plot(predictionThresholds,predAcGT,'r:*','LineWidth',5.0,'MarkerSize',10.0);
    %plot(predictionThresholds,recPrecision,'g:+','LineWidth',2);
    %legend('Forecasting precision');
    %plot(predictionThresholds,predAc,'b--o','DisplayName',strcat('Forecasting accuracy (on recognized) maxSpread:',num2str(maxSpread)),'LineWidth',2);
    %plot(predictionThresholds,predAcGT,'r:*','DisplayName',strcat('Forecasting accuracy (on GROUND TRUTH) maxSpread:',num2str(maxSpread)),'LineWidth',2);
    %plot(predictionThresholds,recPrecision,'g:+','DisplayName','Recognition precision','LineWidth',2);
    axis([gca],[0.0 1.0 0.0 1.0]);
    set(gca,'LineWidth',5.0);
    %legend('-DynamicLegend');
    plot(predictionThresholds,predictionThresholds,'LineWidth',4.0);%,'DisplayName','f(x)=x');
    %legend('Precision (on recognized)', 'Precision (on ground truth)', 'f(x)=x','Location','northwest');%,'Orientation','horizontal');
    legend('Precision', 'f(x)=x','Location','northwest');%,'Orientation','horizontal');
    %legend('Precision (on recognized)', 'f(x)=x','Location','northwest');%,'Orientation','horizontal');
    %legend('-DynamicLegend');
    %title('Prediction Accuracy (Recognized vs Ground Truth)');
    %title('Precision (all states)');
    xlabel('Prediction threshold');
    %ylabel('Percentage (-1 means no predictions produced)');
    ylabel('Precision score');
    set(gca,'FontSize',46);
    set(gca, 'XTick', [0 0.2 0.4 0.6 0.8 1]);
    legend(gca,'show');
    pngFileName = strcat(resultsDir,'/',fnid,'.png');
    saveas(hp,pngFileName,'png')
    
    hsp = figure('Visible','off','Position', [100, 100, 1600, 1200]);
    hold all;
    plot(predictionThresholds,avgSpread,'b--o','LineWidth',5.0,'MarkerSize',10.0);
    plot(predictionThresholds,avgDist,'r:*','LineWidth',5.0,'MarkerSize',10.0);
    %plot(predictionThresholds,recPrecision,'g:+','LineWidth',2);
    %legend('Forecasting precision');
    %plot(predictionThresholds,predAc,'b--o','DisplayName',strcat('Forecasting accuracy (on recognized) maxSpread:',num2str(maxSpread)),'LineWidth',2);
    %plot(predictionThresholds,predAcGT,'r:*','DisplayName',strcat('Forecasting accuracy (on GROUND TRUTH) maxSpread:',num2str(maxSpread)),'LineWidth',2);
    %plot(predictionThresholds,recPrecision,'g:+','DisplayName','Recognition precision','LineWidth',2);
    %axis([gca],[0.0 1.0 0.0 1.0]);
    %legend('-DynamicLegend');
    legend('Average Spread', 'Average Distance', 'northwest');%,'Orientation','horizontal');
    %legend('Precision (on recognized)', 'f(x)=x','Location','northwest');%,'Orientation','horizontal');
    %legend('-DynamicLegend');
    %title('Prediction Accuracy (Recognized vs Ground Truth)');
    %title('Precision (all states)');
    xlabel('Prediction threshold');
    %ylabel('Percentage (-1 means no predictions produced)');
    ylabel('Points');
    set(gca,'FontSize',46);
    legend(gca,'show');
    pngFileName = strcat(resultsDir,'/',fnid,'_sp.png');
    saveas(hsp,pngFileName,'png')
    
    ht = figure('Visible','off','Position', [100, 100, 1600, 1200]);
    hold all;
    plot(predictionThresholds,throughput,'b--o','LineWidth',5.0,'MarkerSize',10.0);
    axis([gca],[0.0 1.0 0.0 200000]);
    legend('Throughput', 'northwest');%,'Orientation','horizontal');
    xlabel('Prediction threshold');
    ylabel('Events/sec');
    set(gca,'FontSize',36);
    legend(gca,'show');
    pngFileName = strcat(resultsDir,'/',fnid,'_through.png');
    saveas(ht,pngFileName,'png')
end

%return;

% row =     predictionThreshold(0)  maxSpread(1)            state(2)
%           predictionsNo(3)        correctPredictionsNo(4) predAc(5)     correctPredictionsNoGT(6) predAcGT(7)
%           allDetected(8)          missed(9)               lossRatio(10) allDetectedGT(11)
%           avgSpread(12)           avgDist(13)

fnidps = strcat(pname,'_seq_',pattern,'_pol_',policy,'_ord_',num2str(order));
%fnidps = strcat('vid_',vid,'_name_',pname,'_seq_',pattern,'_pol_',policy,'_ord_',num2str(order));
csvFileNameps = strcat(resultsDir,'/',fnidps,'_per_state.csv');
resultsps = csvread(csvFileNameps);

pti = 1;
msi = 2;
si = 3;
pni = 4;
cpni = 5;
paci = 6;
cpngti = 7;
pacgti = 8;
lri = 11;
asi = 13;
adi = 14;

states_grid = resultsps(find(resultsps(:,1)==resultsps(1,1) & resultsps(:,2)==resultsps(1,2)),3);
states_grid = states_grid';
%states = [0, 1, 2, 3, 4];
%states = [0, 1, 9, 2, 3, 4, 5, 6, 7]; %fz2
%states = [0, 1, 11, 9, 10, 2, 12, 3, 4, 5, 6, 7]; %fz3
%states = [0, 32, 35, 26, 29, 31, 34, 25, 28, 27, 30, 33, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]; %amin1
%states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]; %amin1
%states = [3, 7, 9, 11, 13, 14, 15, 16, 17, 18, 19, 20]; %amin1
%states = [0, 23, 8, 17, 26, 11, 20, 14, 13, 22, 7, 16, 25, 10, 19, 9, 18, 12, 21, 15, 24, 1, 2, 3, 4, 5]; %amin21
%states = [0, 2, 4, 5]; %amin21
states = [0, 1, 2, 3, 4, 5, 6, 7]; %fz1
states_no = size(states,2);
%states_labels = ['0'; '1'; '9'; '2'; '3'; '4'; '5'; '6'; '7'; '8']; %fz2
states_ticks = 0:(states_no-1);
states_labels = {'0', '1', '2', '3', '4', '5', '6', '7'}; %fz1
%states_labels = {'0', '1', '1^{1}', '2', '3', '4', '5', '6', '7'}; %fz2
%states_labels = {'0', '1', '1^{1}', '1^{2}', '1^{3}', '2', '2^{1}', '3', '4', '5', '6', '7'}; %fz3
%states_labels = {'0', '1', '2', '3', '4',};
%states_labels = {'0', '32', '35', '26', '29', '31', '34', '25', '28', '27', '30', '33', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'}; %amin1
%states_labels = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'}; %amin1
%states_labels = {'3^{te}', '7^{tw}', '9^{tn}', '11^{ts}', '13^{gse}', '14^{gsn}', '15^{gsw}', '16^{gss}', '17^{gen}', '18^{gew}', '19^{ges}', '20^{gee}'}; %amin1
%states_labels = {'0', '23', '8', '17', '26', '11', '20', '14', '13', '22', '7', '16', '25', '10', '19', '9', '18', '12', '21', '15', '24', '1', '2', '3', '4', '5'}; %amin21
%states_labels = {'0', '2', '4', '5'}; %amin21

%{
hs = figure('Visible','on','Position', [100, 100, 1600, 800]);
ax1 = subplot(2,2,1);
ax2 = subplot(2,2,2);
ax3 = subplot(2,2,3);
ax4 = subplot(2,2,4);
hold all;
state_counter = 0;
for state=states
    state_counter = state_counter + 1;
    for maxSpread=maxSpreads
        predAc = zeros(1,thresholdsNo);
        predAcGT = zeros(1,thresholdsNo);
        avgSpread = zeros(1,thresholdsNo);
        avgDist = zeros(1,thresholdsNo);
        for ti=1:thresholdsNo
            threshold = predictionThresholds(ti);
            i = find(resultsps(:,1)==threshold & resultsps(:,2)==maxSpread & resultsps(:,3)==state);
            predAc(ti) = resultsps(i,paci);
            predAcGT(ti) = resultsps(i,pacgti);
            avgSpread(ti) = resultsps(i,asi);
            avgDist(ti) = resultsps(i,adi);
        end
        
        plot(ax1,predictionThresholds,predAc,'DisplayName',strcat('maxSpread:',num2str(maxSpread),'@state',num2str(state)),'LineWidth',2);
        hold(ax1,'on');
        legend(ax1,'-DynamicLegend');
        plot(ax2,predictionThresholds,predAcGT,'DisplayName',strcat('maxSpread:',num2str(maxSpread),'@state',num2str(state)),'LineWidth',2);
        hold(ax2,'on');
        legend(ax2,'-DynamicLegend');
        plot(ax3,predictionThresholds,avgSpread,'DisplayName',strcat('AvgSpread maxSpread:',num2str(maxSpread),'@state',num2str(state)),'LineWidth',2);
        hold(ax3,'on');
        legend(ax3,'-DynamicLegend');
        plot(ax4,predictionThresholds,avgDist,'DisplayName',strcat('AvgDistance maxSpread:',num2str(maxSpread),'@state',num2str(state)),'LineWidth',2);
        hold(ax4,'on');
        legend(ax4,'-DynamicLegend');
    end
end

plot(ax1,predictionThresholds,predictionThresholds,'DisplayName','f(x)=x');
axis([ax1],[0.0 1.0 -1.0 1.0]);
legend(ax1,'-DynamicLegend');
title(ax1,'Prediction Accuracy per state');
xlabel(ax1,'Prediction threshold');
ylabel(ax1,'Percentage (-1 means no predictions produced)');
legend(ax1,'show');

plot(ax2,predictionThresholds,predictionThresholds,'DisplayName','f(x)=x');
axis([ax2],[0.0 1.0 -1.0 1.0]);
legend(ax2,'-DynamicLegend');
title(ax2,'Prediction Accuracy (Ground Truth) per state');
xlabel(ax2,'Prediction threshold');
ylabel(ax2,'Percentage (-1 means no predictions produced)');
legend(ax2,'show');

axis([ax3],[0.0 1.0 -1.0 max(maxSpreads)]);
legend(ax3,'-DynamicLegend');
title(ax3,'Average Spread per state');
xlabel(ax3,'Prediction threshold');
ylabel(ax3,'Points (-1 means no predictions produced)');
legend(ax3,'show');

legend(ax4,'-DynamicLegend');
title(ax4,'Average Distance per state');
xlabel(ax4,'Prediction threshold');
ylabel(ax4,'Points (-1 means no predictions produced)');
legend(ax4,'show');


pngFileNameps = strcat(resultsDir,'/',fnidps,'_per_state.png');
saveas(hs,pngFileNameps,'png');
%}

for maxSpread=maxSpreads
    predAcGrid = zeros(thresholdsNo,states_no);
    recallGrid = zeros(thresholdsNo,states_no);
    predAcGTGrid = zeros(thresholdsNo,states_no);
    avgSpreadGrid = zeros(thresholdsNo,states_no);
    avgDistGrid = zeros(thresholdsNo,states_no);
    state_counter = 0;
    %for si=1:states_no
    for state=states
        state_counter = state_counter + 1;
        for ti=1:thresholdsNo
            threshold = predictionThresholds(ti);
            %state = states(si);
            si = state_counter;
            i = find(resultsps(:,1)==threshold & resultsps(:,2)==maxSpread & resultsps(:,3)==state);
            predAcTmp = resultsps(i,paci);
            if (predAcTmp == -1.0)
                predAcGrid(ti,si) = -1.0;
            else
                predAcGrid(ti,si) = 100.0 * predAcTmp;
            end
            lossRatioTmp = resultsps(i,lri);
            if (lossRatioTmp==-1.0)
                recallGrid(ti,si) = lossRatioTmp;
            else
                recallGrid(ti,si) = 100.0 * (1.0 - lossRatioTmp);
            end
            predAcGtTmp = resultsps(i,pacgti);
            if (predAcGtTmp == -1.0)
                predAcGTGrid(ti,si) = predAcGtTmp;
            else
                predAcGTGrid(ti,si) = 100.0 * predAcGtTmp;
            end
            %avgSpreadGrid(ti,si) = resultsps(i,asi);
            avgSpreadTmp = resultsps(i,asi);
            if (avgSpreadTmp==-1.0)
                avgSpreadGrid(ti,si) = -1;
            else
                avgSpreadGrid(ti,si) = resultsps(i,asi);
            end
            avgDistTmp = resultsps(i,adi);
            if (avgDistTmp==-1.0)
                avgDistGrid(ti,si) = -1;
            else
                %avgDistGrid(ti,si) = resultsps(i,adi);
                avgDistGrid(ti,si) = avgDistTmp - floor(avgSpreadTmp/2);
            end
        end
    end
    [XGrid,YGrid] = meshgrid(states_grid,predictionThresholds);
    
    
    h1 = figure('Visible','on','Position', [100, 100, 1600, 1200]);
    %contourf(XGrid,YGrid,predAcGrid);
    %pcolor(XGrid,YGrid,predAcGrid); %shading(gca,'interp');
    clims = [-1 100];
    i1 = imagesc(states_ticks,predictionThresholds,predAcGrid,clims);
    set(gca,'FontSize',46);
    set(gca,'YDir','normal');
    set(gca, 'XTick', states_ticks);
    %a = gca;
    %a.TickLabelInterpreter = 'latex';
    set(gca,'XTickLabel',states_labels);
    %axis([gca],[0 states_no 0.0 1.0 0 100]);
    ax = ancestor(i1, 'axes');
    xrule = ax.XAxis;
    xrule.FontSize = 26;
    %caxis([-1.0 100.0]);
    colormap bone;
    cmap = colormap;
    mycmap = cmap(17:64,:);
    mycmap(1,:) = [0.0, 0.0, 0.0];
    colormap(mycmap);
    hc = colorbar('XTick', [0 20 40 60 80 100], 'colormap',mycmap(2:48,:));
    %set(hc, 'ylim', [-1 100]);
    %xrule.FontSize = 10;
    %cmap = colormap(gca
    %title('Precision');
    xlabel('State');
    ylabel('Prediction Threshold');
    %set(gca,'XTickLabel',states_labels);
    pngFileNameps = strcat(resultsDir,'/',fnidps,'_imagesc_precision.png')
    saveas(h1,pngFileNameps,'png');
    
    %return;
    
    h2 = figure('Visible','off','Position', [100, 100, 1600, 800]);
    %contourf(XGrid,YGrid,predAcGTGrid);
    %pcolor(XGrid,YGrid,predAcGTGrid); %shading(gca,'interp');
    i2 = imagesc(states_ticks,predictionThresholds,predAcGTGrid); 
    set(gca,'FontSize',20);
    set(gca,'YDir','normal'); 
    set(gca, 'XTick', states_ticks);
    set(gca,'XTickLabel',states_labels);
    ax = ancestor(i2, 'axes');
    xrule = ax.XAxis;
    colormap bone;
    caxis([-1.0 100.0]);
    cmap = colormap;
    cmap(1,:) = [0.3, 0, 0.0052];
    colormap(cmap);
    colorbar;
    xrule.FontSize = 10;
    title('Precision (Ground Truth)');
    xlabel('State');
    ylabel('Prediction Threshold');
    %set(gca,'XTickLabel',states_labels);
    pngFileNameps = strcat(resultsDir,'/',fnidps,'_imagesc_precisiongt.png');
    saveas(h2,pngFileNameps,'png');
    
    h3 = figure('Visible','on','Position', [100, 100, 1600, 1200]);
    %contourf(XGrid,YGrid,avgSpreadGrid);
    %pcolor(XGrid,YGrid,avgSpreadGrid); %shading(gca,'interp');
    clims = [-0.3 10];
    i3 = imagesc(states_ticks,predictionThresholds,avgSpreadGrid,clims); 
    set(gca,'FontSize',46);
    set(gca,'YDir','normal');
    set(gca, 'XTick', states_ticks);
    set(gca,'XTickLabel',states_labels);
    ax = ancestor(i3, 'axes');
    xrule = ax.XAxis;
    xrule.FontSize = 26;
    colormap bone;
    cmap = colormap;
    mycmap = cmap(17:64,:);
    mycmap(48,:) = [0.0, 0.0, 0.0];
    colormap(flipud(mycmap));
    hc = colorbar('XTick', [0 2 4 6 8 10], 'colormap',flipud(mycmap(1:47,:)));
    %caxis([0.0 100.0]);
    %colorbar;
    %title('Spread');
    xlabel('State');
    ylabel('Prediction Threshold');
    %set(gca,'XTickLabel',states_labels);
    pngFileNameps = strcat(resultsDir,'/',fnidps,'_imagesc_spread.png')
    saveas(h3,pngFileNameps,'png');
    
    %return;
    
    h4 = figure('Visible','on','Position', [100, 100, 1600, 1200]);
    %contourf(XGrid,YGrid,avgDistGrid);
    %pcolor(XGrid,YGrid,avgDistGrid); %shading(gca,'interp');
    clims = [-0.3 16];
    i4 = imagesc(states_ticks,predictionThresholds,avgDistGrid,clims); 
    set(gca,'FontSize',46);
    set(gca,'YDir','normal');
    set(gca, 'XTick', states_ticks);
    set(gca,'XTickLabel',states_labels);
    ax = ancestor(i4, 'axes');
    xrule = ax.XAxis;
    xrule.FontSize = 26;
    colormap bone;
    %caxis([0.0 100.0]);
    cmap = colormap;
    mycmap = cmap(17:64,:);
    mycmap(1,:) = [0.0, 0.0, 0.0];
    colormap(mycmap);
    hc = colorbar('XTick', [0 5 10 15], 'colormap',mycmap(2:48,:));
    %colorbar;
    %title('Distance');
    xlabel('State');
    ylabel('Prediction Threshold');
    %set(gca,'XTickLabel',states_labels);
    pngFileNameps = strcat(resultsDir,'/',fnidps,'_imagesc_distance.png')
    saveas(h4,pngFileNameps,'png');
    
    return;
    
    h5 = figure('Visible','off','Position', [100, 100, 1600, 800]);
    %contourf(XGrid,YGrid,predAcGrid);
    %pcolor(XGrid,YGrid,predAcGrid); %shading(gca,'interp');
    i5 = imagesc(states_ticks,predictionThresholds,recallGrid); 
    set(gca,'FontSize',20);
    set(gca,'YDir','normal');
    set(gca, 'XTick', states_ticks);
    set(gca,'XTickLabel',states_labels);
    ax = ancestor(i5, 'axes');
    xrule = ax.XAxis;
    colormap bone;
    caxis([-1.0 100.0]);
    cmap = colormap;
    cmap(1,:) = [0.3, 0, 0.0052];
    colormap(cmap);
    colorbar;
    xrule.FontSize = 10;
    %cmap = colormap(gca)
    
    title('Recall');
    xlabel('State');
    ylabel('Prediction Threshold');
    %set(gca,'XTickLabel',states_labels);
    pngFileNameps = strcat(resultsDir,'/',fnidps,'_imagesc_recall.png');
    saveas(h5,pngFileNameps,'png');
end


