package ui

import java.nio.file.{Files, Paths}
import fsm.CountPolicy
import model.waitingTime.ForecastMethod.ForecastMethod
import CountPolicy.CountPolicy
import model.waitingTime.ForecastMethod

case class WayebConfig(
                        patterns: String = "",
                        declarations: String = "",
                        outputFsm: String = "",
                        outputMc: String = "",
                        outputSpst: String = "",
                        statsFile: String = "",
                        order: Int = ConfigUtils.defaultOrder,
                        confidenceThreshold: Double = ConfigUtils.defaultConfidenceThreshold,
                        maxSpread: Int = ConfigUtils.defaultMaxSpread,
                        horizon: Int = ConfigUtils.defaultHorizon,
                        spreadMethod: ForecastMethod = ConfigUtils.defaultSpreadMethod,
                        maxNoStates: Int = ConfigUtils.maxNoStates,
                        pMin: Double = ConfigUtils.defaultPMin,
                        alpha: Double = ConfigUtils.defaultAlpha,
                        gammaMin: Double = ConfigUtils.defaultGammaMin,
                        r: Double = ConfigUtils.defaultR,
                        verbose: Boolean = ConfigUtils.defaultVerbose,
                        debug: Boolean = ConfigUtils.defaultDebug,
                        streamFile: String = "",
                        domainSpecificStream: String = "",
                        streamArgs: String = "",
                        policy: CountPolicy = ConfigUtils.defaultPolicy,
                        expirationTime: Int = ConfigUtils.defaultExpiration,
                        modelType: String = ConfigUtils.defaultModelType,
                        fsmFile: String = "",
                        mcFile: String = "",
                        experimentsDomain: String = "cards",
                        task: String = "none",
                        pythonPort:Int = 0,
                        javaPort:Int = 0
)

object WayebCLI {
  def main(args: Array[String]): Unit = {

    val countPolicies = ConfigUtils.countPolicies
    val modelTypes = ConfigUtils.modelTypes
    val spreadMethods = ConfigUtils.spreadMethods
    var pythonPort = 0
    var javaPort = 0

    val parser = new scopt.OptionParser[WayebConfig]("wayeb") {
      head("Wayeb", "0.3")

      help("help").text("prints this usage text")

      cmd("compile").
        action((_, c) => c.copy(task = "compile")).
        text("Compile and disambiguate FSM up to an order.").
        children(
          opt[String]("patterns").required().valueName("<file path>").
            action((x, c) => c.copy(patterns = x)).
            validate(x =>
              if (Files.exists(Paths.get(x))) success
              else failure("Pattern file does not exist")).
            text("The XML/SRE file for the patterns (required)."),
          opt[String]("outputFsm").required().valueName("<file path>").
            action((x, c) => c.copy(outputFsm = x)).
            text("Output file for the compiled/disambiguated FSM."),
          opt[String]("declarations").valueName("<file path>").
            action((x, c) => c.copy(declarations = x)).
            validate(x =>
              if (Files.exists(Paths.get(x))) success
              else failure("Declarations file does not exist")).
            text("File declarations (if any)."),
          opt[String]("countPolicy").valueName(countPolicies.toString()).
            action((x, c) => c.copy(policy = CountPolicy.str2Pol(x))).
            validate(x =>
              if (countPolicies.contains(x)) success
              else failure("Count policy should be one of " + countPolicies)).
            text("Counting policy.")
        )

      cmd("mle").
        action((_, c) => c.copy(task = "mle")).
        text("Estimate transition matrix of PMC.").
        children(
          opt[String]("fsm").required().valueName("<file path>").
            action((x, c) => c.copy(fsmFile = x)).
            validate(x =>
              if (Files.exists(Paths.get(x))) success
              else failure("FSM file does not exist")).
            text("FSM (serialized) to be used."),
          opt[String]("stream").required().valueName("<file path>").
            action((x, c) => c.copy(streamFile = x)).
            validate(x =>
              if (Files.exists(Paths.get(x))) success
              else failure("Stream file does not exist")).
            text("The input file with the stream of events (required)."),
          opt[String]("domainSpecificStream").valueName("<file path>").
            action((x, c) => c.copy(domainSpecificStream = x)).
            text("Specify the domain stream."),
          opt[String]("streamArgs").valueName("<file path>").
            action((x, c) => c.copy(streamArgs = x)).
            text("Specify the domain stream arguments (comma separated)."),
          opt[String]("outputMc").required().valueName("<file path>").
            action((x, c) => c.copy(outputMc = x)).
            text("Output file for the calculated Markov Chain.")
        )

      cmd("forecasting").
        action((_, c) => c.copy(task = "forecasting")).
        text("Process stream given a FSM and a learnt PMC (recognition and forecasting).").
        children(
          opt[String]("modelType").required().valueName(modelTypes.toString()).
            action((x, c) => c.copy(modelType = x)).
            validate(x =>
              if (modelTypes.contains(x)) success
              else failure("FSM type should be one of " + modelTypes)).
            text("FSM type."),
          opt[String]("fsm").required().valueName("<file path>").
            action((x, c) => c.copy(fsmFile = x)).
            validate(x =>
              if (Files.exists(Paths.get(x))) success
              else failure("FSM file does not exist")).
            text("FSM (serialized) to be used."),
          opt[String]("mc").valueName("<file path>").
            action((x, c) => c.copy(mcFile = x)).
            validate(x =>
              if (Files.exists(Paths.get(x))) success
              else failure("MC file does not exist")).
            text("Markov chain (serialized) to be used (only for FMMs)."),
          opt[String]("stream").required().valueName("<file path>").
            action((x, c) => c.copy(streamFile = x)).
            validate(x =>
              if (Files.exists(Paths.get(x))) success
              else failure("Stream file does not exist")).
            text("The input file with the stream of events (required)."),
          opt[String]("domainSpecificStream").valueName("<file path>").
            action((x, c) => c.copy(domainSpecificStream = x)).
            text("Specify the domain stream."),
          opt[String]("streamArgs").valueName("<file path>").
            action((x, c) => c.copy(streamArgs = x)).
            text("Specify the domain stream arguments (comma separated)."),
          opt[String]("statsFile").required().valueName("<file path>").
            action((x, c) => c.copy(statsFile = x)).
            text("Output file for statistics."),
          opt[Double]("threshold").valueName("Double >0 <=1.0").
            action((x, c) => c.copy(confidenceThreshold = x)).
            text("Forecasts produced should have probability above this threshold."),
          opt[Int]("maxSpread").valueName("Int >0").
            action((x, c) => c.copy(maxSpread = x)).
            text("Maximum spread allowed for a forecast."),
          opt[Int]("horizon").valueName("Int >0").
            action((x, c) => c.copy(horizon = x)).
            text("Horizon is the \"length\" of the waiting-time distributions, i.e., " +
              "for how may points into the future we want to calculate the completion probability."),
          opt[String]("spreadMethod").valueName(spreadMethods.toString()).
            action((x, c) => c.copy(spreadMethod = ForecastMethod.string2method(x))).
            validate(x =>
              if (spreadMethods.contains(x)) success
              else failure("Spread method should be one of " + spreadMethods)).
            text("Spread method.")
        )

      cmd("recognition").
        action((_, c) => c.copy(task = "recognition")).
        text("Process stream given a FSM (recognition only).").
        children(
          opt[String]("fsm").required().valueName("<file path>").
            action((x, c) => c.copy(fsmFile = x)).
            validate(x =>
              if (Files.exists(Paths.get(x))) success
              else failure("FSM file does not exist: " + x)).
            text("FSM (serialized) to be used."),
          opt[String]("stream").required().valueName("<file path>").
            action((x, c) => c.copy(streamFile = x)).
            validate(x =>
              if (Files.exists(Paths.get(x))) success
              else failure("Stream file does not exist")).
            text("The input file with the stream of events (required)."),
          opt[String]("domainSpecificStream").valueName("<file path>").
            action((x, c) => c.copy(domainSpecificStream = x)).
            text("Specify the domain stream."),
          opt[String]("streamArgs").valueName("<file path>").
            action((x, c) => c.copy(streamArgs = x)).
            text("Specify the domain stream arguments (comma separated)."),
          opt[String]("statsFile").required().valueName("<file path>").
            action((x, c) => c.copy(statsFile = x)).
            text("Output file for statistics.")
        )

      cmd("learnSPST").
        action((_, c) => c.copy(task = "learnSPST")).
        text("Learn symbolic probabilistic suffix automaton.").
        children(
          opt[String]("patterns").required().valueName("<file path>").
            action((x, c) => c.copy(patterns = x)).
            validate(x =>
              if (Files.exists(Paths.get(x))) success
              else failure("Pattern file does not exist")).
            text("The SRE file for the patterns (required)."),
          opt[Int]("pMin").valueName("pMin>0 and pMin<1.0").
            action((x, c) => c.copy(pMin = x)).
            text("This is the symbol threshold. Symbols with lower probability are discarded."),
          opt[Int]("alpha").valueName("alpha>0 and alpha<1.0").
            action((x, c) => c.copy(alpha = x)).
            text("Used to calculate the conditional threshold = (1 + alpha) * gammaMin.\n" +
              " The conditional on the expanded context must be greater than this threshold."),
          opt[Int]("gammaMin").valueName("gammaMin>0 and gammaMin<1.0").
            action((x, c) => c.copy(gammaMin = x)).
            text("Used to calculate the conditional threshold = (1 + alpha) * gammaMin.\n" +
              " The conditional on the expanded context must be greater than this threshold.\n" +
              "Also used for smoothing."),
          opt[Int]("r").valueName("r>0").
            action((x, c) => c.copy(r = x)).
            text("This is the likelihood ratio threshold.\n" +
              "Contexts are expanded if the probability ratio of the conditional on the expanded context by the\n " +
              " conditional on the original context is greater than this threshold."),
          opt[String]("declarations").valueName("<file path>").
            action((x, c) => c.copy(declarations = x)).
            validate(x =>
              if (Files.exists(Paths.get(x))) success
              else failure("Declarations file does not exist")).
            text("File declarations (if any)."),
          opt[String]("stream").required().valueName("<file path>").
            action((x, c) => c.copy(streamFile = x)).
            validate(x =>
              if (Files.exists(Paths.get(x))) success
              else failure("Stream file does not exist")).
            text("The input file with the training stream of events (required)."),
          opt[String]("domainSpecificStream").valueName("<file path>").
            action((x, c) => c.copy(domainSpecificStream = x)).
            text("Specify the domain stream."),
          opt[String]("streamArgs").valueName("<file path>").
            action((x, c) => c.copy(streamArgs = x)).
            text("Specify the domain stream arguments (comma separated)."),
          opt[String]("outputSpst").required().valueName("<file path>").
            action((x, c) => c.copy(outputSpst = x)).
            text("Output file for the SPST.")
        )

      cmd("experiments").
        action((_, c) => c.copy(task = "experiments")).
        text("Run experiments.").
        children(
          opt[String]("domain").required().
            action((x, c) => c.copy(experimentsDomain = x)).
            text("The domain for the experiments, cards or maritime (required).")
        )

      cmd("server").
        action((_, c) => c.copy(task = "server")).
        text("Run server.")

      cmd("service").
        action((_, c) => c.copy(task = "service")).
        text("Run Wayeb as a OOF service.").
        children(
          opt[String]("pythonPort").required().valueName("<port>").
          action((x, c) => c.copy(pythonPort = x.toInt)).
            text("Python port is required."),
          opt[String]("javaPort").required().valueName("<port>").
            action((x,c) => c.copy(javaPort = x.toInt)).
            text("javaPort is required."),
//          opt[String]("patterns").required().valueName("<file path>").
//            action((x, c) => c.copy(patterns = x)).
//            validate(x =>
//              if (Files.exists(Paths.get(x))) success
//              else failure("Pattern file does not exist")).
//            text("The SRE file for the patterns (required)."),
//          opt[String]("initialModel").required().valueName("<file path>").
//            action((x, c) => c.copy(patterns = x)).
//            text("Initial model is required."),
        )

    }

    parser.parse(args, WayebConfig()) match {
      case Some(config) => runWayeb(config)
      case None => throw new IllegalArgumentException("Something is wrong with the arguments.")
    }

  }

  private def runWayeb(config: WayebConfig): Unit = {
    println(ConfigUtils.wayebLogo)
    config.task match {
      case "compile" => BeepBeep.runFSMDisambiguation(config)
      case "mle" => BeepBeep.runMatrixEstimation(config)
      case "forecasting" => BeepBeep.runForecasting(config)
      case "recognition" => BeepBeep.runRecognition(config)
      case "learnSPST" => BeepBeep.runLearnSPST(config)
      case "experiments" => BeepBeep.runExperiments(config)
      case "server" => BeepBeep.runServer(config)
      case "service" => BeepBeep.runEngineService(config)
      case _ => throw new IllegalArgumentException("Unrecognized task.")
    }
  }

}
