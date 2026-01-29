For a description of the options, run:

$ java -jar wayeb-0.6.0-SNAPSHOT.jar --help

For running forecasting with the provided model (water_level.spst), run the following: 

$ java -jar wayeb-0.6.0-SNAPSHOT.jar forecasting --fsm:water_level.spst  --modelType:vmm  --stream:kafka --kafkaConfIn:kafkaIn.properties --domainSpecificStream:forward  --statsFile:kafka --kafkaConfOut:kafkaOut.properties --foreMethod:classify-win --horizon:15 --center:3 --maxSpread:5 --threshold:0.5 --finalsEnabled:true

Change settings in kafka configuration files, if yours are different.

You need to first (before running forecasting) dump the contents of the csv file into the kafka input topic. You may also want to add an extra "terminate" string at the end, if you want wayeb to actually terminate after reading all of the input records.

Each input record is of the following format:

timestamp,eventType,
observedAttribute1Name,observedAttribute1Type,observedAttribute1Value,observedAttribute2Name,observedAttribute2Type,observedAttribute2Value,...|
futureAttribute1NameForTPlus1,futyreAttribute1TypeForTPlus1,futureAttribute1ValueForTPlus1...|
futureAttribute1NameForTPlus2,futyreAttribute1TypeForTPlus2,futureAttribute1ValueForTPlus2...

Everythin before the first | concerns the actual, observed event at time "timestamp". For each atribute of the observed event, we need to specify its name, its type (double or string) and its actual value.

After the first | we have (if any) future events. You do not need to care about these now.

For example, an entry like:

12,WaterLevel,level:double:-0.5732269976035714|level:double:-0.5732269976035714|level:double:-0.5732269976035714

means that we have observed an event of type "WaterLevel" at time 12 and its paylod is a single attribute, named "level", whose type is double and values is -0.5732269976035714. 

We have also produce two future values for t+1 and t+2 (i.e., 13, 14) where the attribut value remains the same.




