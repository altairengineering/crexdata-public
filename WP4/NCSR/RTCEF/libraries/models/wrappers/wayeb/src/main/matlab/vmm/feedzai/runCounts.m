setname = 'Counts';

resultsDir = '/home/zmithereen/data/feedzai/results/';
resultsFiles = {strcat(resultsDir,'increasingG1H8MeanClassifyNextK.csv'),strcat(resultsDir,'increasingG1H8SDFAClassifyNextK.csv'),strcat(resultsDir,'increasingG1H8SPSAClassifyNextK.csv'),strcat(resultsDir,'increasingG1H8SPSTClassifyNextK.csv')};
orderSets = {[-1],[0 1 2], [1 2], [2 3 4]};
prefices = {'MEAN','FMM', 'VMM','PST'};
thresholds = [0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0];



for threshold=thresholds
    for which=[2 3 4]
        orders = orderSets{which};
        resultsFile = resultsFiles{which};
        results = csvread(resultsFile,1,0);
        prefix = strcat(setname,prefices{which});
        counts(threshold,orders,results,prefix,resultsDir)
    end
end
