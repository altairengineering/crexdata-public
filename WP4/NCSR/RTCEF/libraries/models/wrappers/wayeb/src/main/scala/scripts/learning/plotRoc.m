home = getenv('WAYEB_HOME');
fn = strcat(home, '/results/bio/statsRoc.csv');
results = csvread(fn);
pointsPerLine = (size(results,2)-2)/2;


markers = {'o','*','+','x','s','d'};
styles = {'-','--',':','-.'};

figure('units','normalized','outerposition',[0 0 1 1],'visible','on');
legends = {};
prevDist = -1;
for line=1:size(results,1)
    recalls = results(line,3:3+pointsPerLine-1);
    fallouts = results(line,3+pointsPerLine:3+pointsPerLine+pointsPerLine-1);
    bins = results(line,1);
    distance = results(line,2);
    lineLegend = strcat('bins=',num2str(bins),'dist=',num2str(distance));
    if (prevDist == -1)
        markerIndex = 1;
        styleIndex = 1;
    else
        if (prevDist ~= distance)
            markerIndex = markerIndex + 1;
            styleIndex = styleIndex + 1;
        end    
    end
    prevDist = distance;
    plot(fallouts,recalls,'Linewidth',3,'MarkerSize',15,'Marker',markers{markerIndex},'LineStyle',styles{styleIndex});
    hold on;
    legends{1,line} = lineLegend; 
end
hold off;
ylabel('recall');
xlabel('fallout');
legend(legends, 'location', 'southeast');
set(gca, 'YLim',[0,1]);
set(gca, 'XLim',[0,1]);
set(gcf,'Color','w');
set(gca,'FontSize',26);
grid on;
grid minor;