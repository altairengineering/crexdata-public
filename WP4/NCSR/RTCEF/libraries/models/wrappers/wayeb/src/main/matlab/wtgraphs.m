fn_wts = '/home/zmithereen/ownCloud/all0_wts.csv';
fn_preds = '/home/zmithereen/ownCloud/all0_predictions.csv';
results_wt = csvread(fn_wts);
results_preds = csvread(fn_preds);
states_no = size(results_wt,2) - 1;
state = 5;
timepoints = results_wt(:,1);
wt = results_wt(:,state+1);
marker = '*';
plot(timepoints,wt,'Marker',marker,'LineWidth',2.0,'MarkerSize',2.0);

for p=1:9
   start = results_preds((p-1)*states_no + state,3);
   finish = results_preds((p-1)*states_no + state,4);
   prob = results_preds((p-1)*states_no + state,5);
   hold on;
   ymax = 0.0480;
   yoffset = (p-1)*0.0015;
   plot([start,finish],[ymax-yoffset,ymax-yoffset],'--x','LineWidth',3.0,'MarkerSize',20.0);
end;
axis([gca],[-1.0 200.0 0.0 ymax]);
legend('WT Distribution','threshold=0.1','threshold=0.2','threshold=0.3','threshold=0.4','threshold=0.5','threshold=0.6','threshold=0.7','threshold=0.8','threshold=0.9');
xlabel('Number of future events');
ylabel('Completion probability');
set(gca,'FontSize',26);
set(gcf, 'Color', 'w');

%plot([start(2),end1(2)],[0.005,0.005],'--x','LineWidth',2.0,'MarkerSize',10.0,'DisplayName','interval:')