# Complex Event Forecasting with variable-oder Markov models

If you want to run forecasting with variable-order Markov models,
the first step is to learn a model for a given pattern from a given training stream.
You can do this by running the following command:
````
$ java -jar wayeb-0.3.0-SNAPSHOT.jar learnSPST --patterns:$WAYEB_HOME/patterns/maritime/port/pattern.sre --declarations:$WAYEB_HOME/patterns/maritime/port/declarationsDistance1.sre --stream:$WAYEB_HOME/data/maritime/227592820.csv --domainSpecificStream:maritime --streamArgs: --outputSpst:$WAYEB_HOME/results/tmp.spst
````
You must specify:
* the path to the file with the patterns;
* the path to the declarations file;
* the path to the file with the training stream;
* the stream domain in case of csv files 
(whose values should simply be `json` in case of JSON files);
* `streamArgs:` is to be used if you need to pass any special arguments to the event stream parser;
* the path to the learnt prediction suffix tree that is going to be serialized.

You can then run forecasting with the following command:
````
$ java -jar wayeb-0.3.0-SNAPSHOT.jar forecasting --modelType:vmm --fsm:$WAYEB_HOME/results/tmp.spst --stream:$WAYEB_HOME/data/maritime/227592820.csv --domainSpecificStream:maritime --streamArgs: --statsFile:$WAYEB_HOME/results/forestats
````
You must specify:
* the model type (fmm or vmm). vmm in this case for the variable-order Markov model (prediction suffix tree) we learnt.
* the path to the learnt serialized prediction suffix tree.
* the path to the file with the test stream;
* the stream domain in case of csv files 
(whose values should simply be `json` in case of JSON files);
* `streamArgs:` is to be used if you need to pass any special arguments to the event stream parser;
* the path to the file with the generated statistics.

Note that here we use tha same stream both for training and testing.
This is done for convenience.
You should normally use different streams.

See also the script [ui.demo.RunCLI](src/main/scala/ui/demo/RunCLI.scala).