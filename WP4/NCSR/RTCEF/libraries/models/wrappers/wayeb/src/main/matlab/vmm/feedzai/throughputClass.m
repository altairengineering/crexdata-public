function [data1, data2] = trainTimeClass(minDistances, maxDistances, orders, maxSpread, results)

    groupsNo = size(minDistances,2);
    stacksPerGroup = size(orders,2);
    data1 = zeros(groupsNo,stacksPerGroup);
    data2 = zeros(groupsNo,stacksPerGroup);

    for g=1:groupsNo
        minDistance = minDistances(g);
        maxDistance = maxDistances(g);
        for s=1:stacksPerGroup
            order = orders(s);
            index = find(results(:,1)==order & results(:,2)==maxSpread & results(:,3)==minDistance & results(:,4)==maxDistance);
            throughput = mean(results(index,18));
            states = mean(results(index,16));
            data1(g,s) = throughput;
            data2(g,s) = states;
         end
    end
end