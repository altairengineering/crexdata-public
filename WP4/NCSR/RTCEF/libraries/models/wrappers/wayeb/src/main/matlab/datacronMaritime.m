home = '/home/zmithereen/';
pattern = 'fishing';
%fna0 = strcat(home,'ownCloud/',pattern,'_features_all_m_0.stats_gathered.csv');
%resultsa0 = csvread(fna0);
fna1 = strcat(home,'ownCloud/',pattern,'_features_all_m_1.stats_gathered.csv');
resultsa1 = csvread(fna1);
%fna2 = strcat(home,'ownCloud/',pattern,'_features_all_m_2.stats_gathered.csv');
%resultsa2 = csvread(fna2);

%fnn0 = strcat(home,'ownCloud/',pattern,'_features_none_m_0.stats_gathered.csv');
%resultsn0 = csvread(fnn0);
fnn1 = strcat(home,'ownCloud/',pattern,'_features_none_m_1.stats_gathered.csv');
resultsn1 = csvread(fnn1);
%fnn2 = strcat(home,'ownCloud/',pattern,'_features_none_m_2.stats_gathered.csv');
%resultsn2 = csvread(fnn2);
%fnn3 = strcat(home,'ownCloud/',pattern,'_features_none_m_3.stats_gathered.csv');
%resultsn3 = csvread(fnn3);
%fnn4 = strcat(home,'ownCloud/',pattern,'_features_none_m_4.stats_gathered.csv');
%resultsn4 = csvread(fnn4);

variations = 2;

thresholds = resultsa1(:,1);
et = zeros(size(thresholds,1),variations);
prec = zeros(size(thresholds,1),variations);
spread = zeros(size(thresholds,1),variations);
dist = zeros(size(thresholds,1),variations);

%et(:,1) = resultsn0(:,2)/1000000000.0;
et(:,1) = resultsn1(:,2)/1000000000.0;
%et(:,3) = resultsn2(:,2)/1000000000.0;
%et(:,4) = resultsn3(:,2)/1000000000.0;
%et(:,5) = resultsn4(:,2)/1000000000.0;
%et(:,6) = resultsa0(:,2)/1000000000.0;
et(:,2) = resultsa1(:,2)/1000000000.0;
%et(:,6) = resultsa2(:,2)/1000000000.0;

%prec(:,1) = resultsn0(:,3);
prec(:,1) = resultsn1(:,3);
%prec(:,3) = resultsn2(:,3);
%prec(:,4) = resultsn3(:,3);
%prec(:,5) = resultsn4(:,3);
%prec(:,6) = resultsa0(:,3);
prec(:,2) = resultsa1(:,3);
%prec(:,6) = resultsa2(:,3);

%spread(:,1) = resultsn0(:,4);
spread(:,1) = resultsn1(:,4);
%spread(:,3) = resultsn2(:,4);
%spread(:,4) = resultsn3(:,4);
%spread(:,5) = resultsn4(:,4);
%spread(:,6) = resultsa0(:,4);
spread(:,2) = resultsa1(:,4);
%spread(:,6) = resultsa2(:,4);

%dist(:,1) = resultsn0(:,5);
dist(:,1) = resultsn1(:,5);
%dist(:,3) = resultsn2(:,5);
%dist(:,4) = resultsn3(:,5);
%dist(:,5) = resultsn4(:,5);
%dist(:,6) = resultsa0(:,5);
dist(:,2) = resultsa1(:,5);
%dist(:,6) = resultsa2(:,5);

lines = [1 2];

figure;
for i = lines
    plot(thresholds,et(:,i),'--*','LineWidth',3.0);
    hold on;
end
ylabel('Execution time (sec)');
xlabel('Confidence threshold');
legend('No features, m=0','No features, m=1','No features, m=2','All features, m=0','All features, m=1','All features, m=2');
%legend('No features, m=1','No features, m=2','All features, m=1','All features, m=2');
set(gca,'FontSize',32);
set(gcf, 'Color', 'w');
hold off;

collapses = ones(size(thresholds,1)) .* -1.0;

figure;
for i=lines
    plot(thresholds,prec(:,i),'--*','LineWidth',3.0);
    hold on;
end
plot(thresholds,thresholds,'LineWidth',4.0);
plot(thresholds,collapses,'red','LineWidth',4.0);
ylabel('Precision');
xlabel('Confidence threshold');
legend('No features, m=0','No features, m=1','No features, m=2','All features, m=0','All features, m=1','All features, m=2','f(x)=x','Collapsed');
%legend('No features, m=1','No features, m=2','All features, m=1','All features, m=2','f(x)=x','Collapsed');
set(gca,'FontSize',32);
set(gcf, 'Color', 'w');
hold off;

figure;
for i=lines
    plot(thresholds,spread(:,i),'--*','LineWidth',3.0);
    hold on;
end
plot(thresholds,collapses,'red','LineWidth',4.0);
ylabel('Spread (minutes)');
xlabel('Confidence threshold');
legend('No features, m=0','No features, m=1','No features, m=2','All features, m=0','All features, m=1','All features, m=2','Collapsed');
%legend('No features, m=1','No features, m=2','All features, m=1','All features, m=2','Collapsed');
set(gca,'FontSize',32);
set(gcf, 'Color', 'w');
hold off;

figure;
for i=lines
    plot(thresholds,dist(:,i),'--*','LineWidth',3.0);
    hold on;
end
plot(thresholds,collapses,'red','LineWidth',4.0);
ylabel('Distance (minutes)');
xlabel('Confidence threshold');
legend('No features, m=0','No features, m=1','No features, m=2','All features, m=0','All features, m=1','All features, m=2','Collapsed');
%legend('No features, m=1','No features, m=2','All features, m=1','All features, m=2','Collapsed');
set(gca,'FontSize',32);
set(gcf, 'Color', 'w');
hold off;
