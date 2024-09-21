import getopt
import sys

from services import (collector, engine, factory, observer, optimiser, replayer, reader)

HELP = "main.py -s <service> -c <config path>\n" + \
       "Available services:\n" + \
       " - collector: collects data from input stream\n" + \
       " - engine: performs CEP/R\n" + \
       " - factory: train/tests models\n" + \
       " - observer: observes metrics\n" + \
       " - optimiser: instructs factory and optmises parameters\n" + \
       " - replayer: replays the dataset\n"


def main(argv):
    opts, _ = getopt.getopt(argv, "hs:c:",
                            ["service=", "config="])
    service = None
    config_path = None

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(HELP)
            sys.exit()
        elif opt in ("-s", "--service"):
            service = arg
        elif opt in ("-c", "--config"):
            config_path = arg
        else:
            print("Wrong arguments. \n" + HELP)
            sys.exit()
    if service not in ("collector", "engine", "factory", "observer",
                       "optimiser", "replayer", "reader"):
        print(HELP)
        sys.exit()

    if service == "optimiser":
        opt_service = optimiser.Optimiser(config_path)
        opt_service.deploy()
    elif service == "engine":
        engine_service = engine.Engine(config_path)
        engine_service.deploy(no_loop=True)
    elif service == "collector":
        collector_service = collector.Collector(config_path)
        collector_service.deploy()
    elif service == "observer":
        observer_service = observer.Observer(config_path)
        observer_service.deploy()
    elif service == "factory":
        factory_service = factory.Factory(config_path)
        factory_service.deploy()
    elif service == "replayer":
        replayer_service = replayer.Replayer(config_path)
        replayer_service.deploy(no_loop=True)
    elif service == "reader":
        reader_service = reader.Reader(config_path)
        reader_service.deploy()
    else:
        raise Exception(f"Service {service} does not exist here.")


if __name__ == "__main__":
    main(sys.argv[1:])
