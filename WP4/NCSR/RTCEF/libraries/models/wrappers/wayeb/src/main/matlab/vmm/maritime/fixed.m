function scores = fixed(maxSpread, threshold, orders, results, column)
    distances = [60 120 180];
    scores = zeros(size(orders,2),1);

    o = 0;
    for order=orders
        o = o + 1;
        %scoresForOrder = zeros(size(distances,2),1);
        i = 0;
        for distance=distances
            i = i + 1;
            index = find(results(:,1)==order & results(:,2)==maxSpread & results(:,3)==distance & results(:,4)==distance & results(:,5)==threshold);
            %scoresForOrder(i) = results(index,7);
            scores(o,i) = results(index,column);
        end
        %scores{o,1} = scoresForOrder;
    end
end