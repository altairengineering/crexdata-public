

resultsDir = '/mnt/Warehouse/wayeb/aminess/dump/maxSpread10/';

%cd(resultsDir);
csvFiles = fullfile(resultsDir,'*.csv');

files = dir(csvFiles);

horizon = 200;
maxSpread = 0.1*horizon;

for file = files'
    fn = file.name;
    fn_noext0 = strsplit(fn,'.');
    fn_noext1 = fn_noext0(1);
    fn_noext = fn_noext1{1,1};
    fn_split = strsplit(fn_noext,'_');
    vid0 = fn_split(1);
    pattern0 = fn_split(2);
    %method0 = fn_split(3);
    vid = vid0{1,1};
    pattern = pattern0{1,1};
    %method = method0{1,1};
    
    ssi = 1;
    mni = 2;
    eti = 3;
    tpi = 4;
    pti = 5;
    thresi = 6;
    pai = 9;
    lri = 12;
    asi = 13;
    adi = 14;
    pari = 19;
    lrri = 22;
    
    m = csvread(strcat(resultsDir,fn));
    
    eventsConsumed = m(1,1);
    matches = m(1,2);
    
    if (matches==0)
        continue;
    end
    
    execTime = m(:,eti)./1000;
    predictTime = m(:,pti)./1000;
    throughput = m (:,tpi);
    thres = m(:,thresi);
    %predictionsNo = m(:,6);
    %correctPredictionsNo = m(:,7);
    predAc = m(:,pai);
    %allDetected = m(:,9);
    %missed = m(:,10);
    %lossRatio = m(:,lri);
    avgSpread = m(:,asi);
    avgDist = m(:,adi);
    predACRand = m(:,pari);
    
    ac2sp = predAc./avgSpread;
    
    h = figure('Visible','off','Position', [100, 100, 1600, 800]);
    descr = {strcat('Pattern: ', pattern, ' Vessel: ', vid, ' Events Consumed: ', num2str(eventsConsumed))};
    
    %axxx = axes('Position',[0 0 1 1],'Visible','off');
    
    ax1 = subplot(2,2,1);
    plot(thres,execTime,thres,predictTime);
    title('Execution and Prediction time');
    xlabel('Prediction threshold');
    ylabel('Time (us)');
    legend('Total execution time', 'Prediction time only');
    
    ax2 = subplot(2,2,2);
    plot(thres,throughput);
    title('Throughput');
    xlabel('Prediction threshold');
    ylabel('Events/sec');
    
    
    
    ax3 = subplot(2,2,3);
    plot(thres,predAc,thres,predACRand,thres,thres);
    %hold on;
    axis([ax3],[0.0 1.0 0.0 1.0]);
    %yyaxis right;
    %plot(thres,ac2sp);
    %yyaxis left;
    title('Prediction Accuracy (Wayeb vs Random)');
    xlabel('Prediction threshold');
    ylabel('Percentage');
    legend('Prediction Accuracy (Wayeb)', 'Prediction Accuracy (Random)', 'f(x)=x',  'Location','northwest');
    
    ax4 = subplot(2,2,4);
    msa = maxSpread*(ones(size(thres,1)));
    plot(thres,avgSpread,thres,avgDist);
    hold on;
    plot(thres,msa,'Color','red');
    axis([ax4],[0.0 1.0 -1 horizon]);
    title('Spread and Distance (avg)');
    xlabel('Prediction threshold');
    ylabel('Timepoints');
    legend('Average Spread', 'Average Distance', 'Max Spread allowed', 'Location','northwest');
    
    %axes(ax4);
    text(0.5,-40.0,descr);  
    %set(gcf,'name',fn_noext,'numbertitle','off');
  
    saveas(gcf,strcat(resultsDir,fn_noext,'.png'),'png');
    
end