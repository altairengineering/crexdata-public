# Running experiments describe in **DBLP:journals/vldbj/AlevizosAP20**

In order to reproduce the experiments described in **DBLP:journals/vldbj/AlevizosAP20** 
(see [How to cite Wayeb](decs/references.md)),
you need to build Wayeb and set the $WAYEB_HOME environment variable,
as described in [Building](docs/building.md).

Then you can run the following command to run the experiments with the synthetic dataset 
simulating credit card transactions.
````
$ java -jar wayeb-0.3.0-SNAPSHOT.jar experiments --domain:cards
````

The following command runs the experiments with the maritime dataset:
````
$ java -jar wayeb-0.3.0-SNAPSHOT.jar experiments --domain:maritime
````

Results are stored under $WAYEB_HOME/results.