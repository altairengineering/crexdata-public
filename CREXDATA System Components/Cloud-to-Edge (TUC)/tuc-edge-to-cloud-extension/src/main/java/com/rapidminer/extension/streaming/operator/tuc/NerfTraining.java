package com.rapidminer.extension.streaming.operator.tuc;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.rapidminer.example.Attribute;
import com.rapidminer.example.table.AttributeFactory;
import com.rapidminer.example.utils.ExampleSetBuilder;
import com.rapidminer.example.utils.ExampleSets;
import com.rapidminer.extension.streaming.ioobject.StreamDataContainer;
import com.rapidminer.extension.streaming.operator.AbstractStreamOperator;
import com.rapidminer.extension.streaming.operator.StreamingNest;
import com.rapidminer.extension.streaming.utility.graph.*;
import com.rapidminer.operator.OperatorDescription;
import com.rapidminer.operator.OperatorException;
import com.rapidminer.operator.UserError;
import com.rapidminer.operator.ports.OutputPort;
import com.rapidminer.parameter.*;
import com.rapidminer.parameter.conditions.AndParameterCondition;
import com.rapidminer.parameter.conditions.EqualTypeCondition;
import com.rapidminer.parameter.conditions.OrParameterCondition;
import com.rapidminer.parameter.conditions.ParameterCondition;
import com.rapidminer.tools.LogService;
import com.rapidminer.tools.Ontology;
import com.rapidminer.tools.container.Pair;

import java.util.Collections;
import java.util.List;
import java.util.logging.Logger;

public class NerfTraining extends AbstractStreamOperator implements StreamSource {
    private static final Logger logger = LogService.getRoot();
    private final OutputPort jsonOutput = getOutputPorts().createPort("output stream 1");
    //common fields
    private static final String PARAMETER_OPERATION_MODE = "operation_mode"; // category: flower / ns3
    private static final String PARAMETER_SEED = "seed";

    private static final String PARAMETER_DATASET = "dataset";
    private static final String PARAMETER_DATASET_PATH = "dataset_path";
    private static final String PARAMETER_DATA_DIRNAME = "data_directory_name";
    private static final String PARAMETER_MASK_DIRNAME = "mask_directory_name";
    private static final String PARAMETER_DOWNSCALE = "downscale";
    private static final String PARAMETER_USE_AMP = "use_AMP";
    private static final String PARAMETER_CHECKPOINT_PATH = "checkpoint_path";
    private static final String PARAMETER_RAY_SAMPLES = "ray_samples";

    // non-common fields
    private static final String PARAMETER_NEAR = "near";
    private static final String PARAMETER_FAR = "far";
    // Rendering/Boundary (Eval & Train)
    private static final String PARAMETER_CHUNK_POINTS = "chunk_points";
    private static final String PARAMETER_BOUNDARY_MARGIN = "boundary_margin";

    // Batching & Data (Eval & Train)
    private static final String PARAMETER_TEST_BATCH_SIZE = "test_batch_size";
    private static final String PARAMETER_SUPPORT_RAYS = "support_rays";

    // Optimization (Eval & Train)
    private static final String PARAMETER_OPTIMIZER = "optimizer";
    private static final String PARAMETER_LR = "lr";
    private static final String PARAMETER_ENCODING_LR = "encoding_lr";
    private static final String PARAMETER_SIGMA_LR = "sigma_lr";
    private static final String PARAMETER_COLOR_LR = "color_lr";
    private static final String PARAMETER_BG_LR = "bg_lr";

    // Checkpoint (Eval & Train)
    private static final String PARAMETER_PREFIX = "prefix";
    private static final String PARAMETER_TTO = "tto";
    private static final String PARAMETER_FNAME = "file_name";
    // Training Step & Batch
    private static final String PARAMETER_SAVE_STEP = "save_step";
    private static final String PARAMETER_BATCH_SIZE = "batch_size";
    private static final String PARAMETER_QUERY_RAYS = "query_rays";

    // Model & Architecture
    private static final String PARAMETER_CELL_DIM = "cell_dim";
    private static final String PARAMETER_NUM_SUBMODULES = "num_submodules";
    private static final String PARAMETER_SIGMA_DEPTH = "sigma_depth";
    private static final String PARAMETER_COLOR_DEPTH = "color_depth";
    private static final String PARAMETER_DIM_HIDDEN = "dim_hidden";
    private static final String PARAMETER_COLOR_HIDDEN = "color_hidden";
    private static final String PARAMETER_LOG2_HASHMAP_SIZE = "log2_hashmap_size";
    private static final String PARAMETER_MAX_RESOLUTION = "max_resolution";
    private static final String PARAMETER_NO_BG_NERF = "no_bg_nerf";
    private static final String PARAMETER_BG_HIDDEN = "bg_hidden";

    // Scheduling & Loop
    private static final String PARAMETER_NO_SCHEDULER = "no_scheduler";
    private static final String PARAMETER_DECAY_FACTOR = "decay_factor";
    private static final String PARAMETER_USE_STORED_ARGS = "use_stored_args";
    private static final String PARAMETER_OUTER_STEPS = "outer_steps";
    private static final String PARAMETER_INNER_ITER = "inner_iterations";
    private static final String PARAMETER_INNER_LR = "inner_lr";


    public NerfTraining(OperatorDescription description) {
        super(description);
        getTransformer().addGenerationRule(jsonOutput, StreamDataContainer.class);
    }

    @Override
    public List<ParameterType> getParameterTypes() {
        List<ParameterType> parameterTypes = super.getParameterTypes();
        ParameterTypeCategory nerfOperationModeParam = new ParameterTypeCategory(
                PARAMETER_OPERATION_MODE,
                "Select the NeRF operation: train, eval, or view.",
                new String[]{"train", "eval", "view"},
                0,  // default index => "train"
                false
        );
        parameterTypes.add(nerfOperationModeParam);
        ParameterCondition trainCondition = new EqualTypeCondition(
                this, PARAMETER_OPERATION_MODE, new String[]{"train"}, false, 0);
        ParameterCondition evalCondition = new EqualTypeCondition(
                this, PARAMETER_OPERATION_MODE, new String[]{"eval"}, false, 1);
        ParameterCondition viewCondition = new EqualTypeCondition(
                this, PARAMETER_OPERATION_MODE, new String[]{"view"}, false, 2);

        // Condition for fields common to Training AND Evaluation
        OrParameterCondition trainOrEvalCondition = new OrParameterCondition(
                this, false, trainCondition, evalCondition);

        // --------------------------------------------------------------------
        // 3. CORE NeRF / DATA PARAMETERS (Common to all 3: train, eval, view)
        // --------------------------------------------------------------------

        ParameterTypeInt seedParam = new ParameterTypeInt(
                PARAMETER_SEED, "Random seed for experiment reproducibility.", 0, Integer.MAX_VALUE, false);
        parameterTypes.add(seedParam);
        ParameterTypeString dataset = new ParameterTypeString(
                PARAMETER_DATASET, "Dataset name.", false);
        parameterTypes.add(dataset);
        ParameterTypeString datasetPath = new ParameterTypeString(
                PARAMETER_DATASET_PATH, "Path to the root data directory (e.g., data/drz/).", false);
        parameterTypes.add(datasetPath);

        ParameterTypeString dataDirname = new ParameterTypeString(
                PARAMETER_DATA_DIRNAME, "Subdirectory name for data (e.g., balanced).", false);
        parameterTypes.add(dataDirname);

        ParameterTypeString maskDirname = new ParameterTypeString(
                PARAMETER_MASK_DIRNAME, "Subdirectory name for masks (e.g., g22_grid_bm110_ss11).", false);
        parameterTypes.add(maskDirname);

        ParameterTypeDouble downscale = new ParameterTypeDouble(
                PARAMETER_DOWNSCALE, "Image downscale factor.", 0.0, 1.0, false);
        parameterTypes.add(downscale);

        ParameterTypeString checkpointPath = new ParameterTypeString(
                PARAMETER_CHECKPOINT_PATH, "Path for saving/loading model weights.", false);
        parameterTypes.add(checkpointPath);

        ParameterTypeBoolean useAMP = new ParameterTypeBoolean(
                PARAMETER_USE_AMP, "Use AMP.", false, false);
        parameterTypes.add(useAMP);

        // --------------------------------------------------------------------
        // 4. EVAL AND TRAIN COMMON PARAMETERS
        // --------------------------------------------------------------------


        ParameterTypeDouble near = new ParameterTypeDouble(
                PARAMETER_NEAR, "Near clipping plane distance.", 0.0, 100.0, false);
        near.registerDependencyCondition(trainOrEvalCondition);
        parameterTypes.add(near);

        ParameterTypeDouble far = new ParameterTypeDouble(
                PARAMETER_FAR, "Far clipping plane distance.", 0.0, Double.MAX_VALUE, false);
        far.registerDependencyCondition(trainOrEvalCondition);
        parameterTypes.add(far);

        ParameterTypeInt raySamples = new ParameterTypeInt(
                PARAMETER_RAY_SAMPLES, "Number of samples per ray (e.g., 96).", 1, 1024, false);
        raySamples.registerDependencyCondition(trainOrEvalCondition); // Note: also used in view, but the value differs.
        parameterTypes.add(raySamples);

        ParameterTypeInt chunkPoints = new ParameterTypeInt(
                PARAMETER_CHUNK_POINTS, "Max points processed in parallel (e.g., 4000000).", 1, Integer.MAX_VALUE, false);
        chunkPoints.registerDependencyCondition(trainOrEvalCondition);
        parameterTypes.add(chunkPoints);

        ParameterTypeDouble boundaryMargin = new ParameterTypeDouble(
                PARAMETER_BOUNDARY_MARGIN, "Margin for the scene bounding box (e.g., 1.0).", 0.0, 100.0, false);
        boundaryMargin.registerDependencyCondition(trainOrEvalCondition);
        parameterTypes.add(boundaryMargin);

        // Batching & Data
        ParameterTypeInt testBatchSize = new ParameterTypeInt(
                PARAMETER_TEST_BATCH_SIZE, "Batch size for testing/evaluation.", 1, Integer.MAX_VALUE, false);
        testBatchSize.registerDependencyCondition(trainOrEvalCondition);
        parameterTypes.add(testBatchSize);

        ParameterTypeInt supportRays = new ParameterTypeInt(
                PARAMETER_SUPPORT_RAYS, "Number of support rays (e.g., 4000).", 1, Integer.MAX_VALUE, false);
        supportRays.registerDependencyCondition(trainOrEvalCondition);
        parameterTypes.add(supportRays);

        // Optimization
        ParameterTypeCategory optimizer = new ParameterTypeCategory(
                PARAMETER_OPTIMIZER, "Optimization algorithm.", new String[]{"adam", "sgd"}, 0, false);
        optimizer.registerDependencyCondition(trainOrEvalCondition);
        parameterTypes.add(optimizer);

        ParameterTypeDouble lr = new ParameterTypeDouble(
                PARAMETER_LR, "Base learning rate.", 0.0, 1.0, false);
        lr.registerDependencyCondition(trainOrEvalCondition);
        parameterTypes.add(lr);

        ParameterTypeDouble encodingLr = new ParameterTypeDouble(
                PARAMETER_ENCODING_LR, "Learning rate for encoding.", 0.0, 1.0, false);
        encodingLr.registerDependencyCondition(trainOrEvalCondition);
        parameterTypes.add(encodingLr);

        ParameterTypeDouble sigmaLr = new ParameterTypeDouble(
                PARAMETER_SIGMA_LR, "Learning rate for density network.", 0.0, 1.0, false);
        sigmaLr.registerDependencyCondition(trainOrEvalCondition);
        parameterTypes.add(sigmaLr);

        ParameterTypeDouble colorLr = new ParameterTypeDouble(
                PARAMETER_COLOR_LR, "Learning rate for color network.", 0.0, 1.0, false);
        colorLr.registerDependencyCondition(trainOrEvalCondition);
        parameterTypes.add(colorLr);

        ParameterTypeDouble bgLr = new ParameterTypeDouble(
                PARAMETER_BG_LR, "Learning rate for background network.", 0.0, 1.0, false);
        bgLr.registerDependencyCondition(trainOrEvalCondition);
        parameterTypes.add(bgLr);

        // Checkpoint
        ParameterTypeString prefix = new ParameterTypeString(
                PARAMETER_PREFIX, "Prefix for checkpoint filenames (e.g., best).", false);
        prefix.registerDependencyCondition(trainOrEvalCondition);
        parameterTypes.add(prefix);


        // --------------------------------------------------------------------
        // 5. EVAL UNIQUE PARAMETERS
        // --------------------------------------------------------------------

        ParameterTypeString tto = new ParameterTypeString(
                PARAMETER_TTO, "Time-to-Output steps/intervals (e.g., 0,1,5).", true); // Optional, but usually provided
        tto.registerDependencyCondition(evalCondition);
        parameterTypes.add(tto);


        // --------------------------------------------------------------------
        // 6. VIEW UNIQUE PARAMETERS
        // --------------------------------------------------------------------

        ParameterTypeString fname = new ParameterTypeString(
                PARAMETER_FNAME, "Output filename for the view rendering (e.g., best).", false);
        fname.registerDependencyCondition(viewCondition);
        parameterTypes.add(fname);


        // --------------------------------------------------------------------
        // 7. TRAIN UNIQUE PARAMETERS
        // --------------------------------------------------------------------

        // Training Steps & Batch
        ParameterTypeInt saveStep = new ParameterTypeInt(
                PARAMETER_SAVE_STEP, "Save checkpoint every N steps (e.g., 1000).", 1, Integer.MAX_VALUE, false);
        saveStep.registerDependencyCondition(trainCondition);
        parameterTypes.add(saveStep);

        ParameterTypeInt batchSize = new ParameterTypeInt(
                PARAMETER_BATCH_SIZE, "Training batch size.", 1, Integer.MAX_VALUE, false);
        batchSize.registerDependencyCondition(trainCondition);
        parameterTypes.add(batchSize);

        ParameterTypeInt queryRays = new ParameterTypeInt(
                PARAMETER_QUERY_RAYS, "Number of query rays (e.g., 2000).", 1, Integer.MAX_VALUE, false);
        queryRays.registerDependencyCondition(trainCondition);
        parameterTypes.add(queryRays);

        ParameterTypeInt cellDim = new ParameterTypeInt(
                PARAMETER_CELL_DIM, "Cell dimension (e.g., 5).", 1, 100, false);
        cellDim.registerDependencyCondition(trainCondition);
        parameterTypes.add(cellDim);

        // Model & Architecture
        ParameterTypeInt numSubmodules = new ParameterTypeInt(
                PARAMETER_NUM_SUBMODULES, "Number of submodules in the model (e.g., 4).", 1, 100, false);
        numSubmodules.registerDependencyCondition(trainCondition);
        parameterTypes.add(numSubmodules);

        ParameterTypeInt sigmaDepth = new ParameterTypeInt(
                PARAMETER_SIGMA_DEPTH, "Depth of the density network (e.g., 2).", 1, 100, false);
        sigmaDepth.registerDependencyCondition(trainCondition);
        parameterTypes.add(sigmaDepth);

        ParameterTypeInt colorDepth = new ParameterTypeInt(
                PARAMETER_COLOR_DEPTH, "Depth of the color network (e.g., 2).", 1, 100, false);
        colorDepth.registerDependencyCondition(trainCondition);
        parameterTypes.add(colorDepth);

        ParameterTypeInt dimHidden = new ParameterTypeInt(
                PARAMETER_DIM_HIDDEN, "Hidden dimension for the density network (e.g., 64).", 1, 1024, false);
        dimHidden.registerDependencyCondition(trainCondition);
        parameterTypes.add(dimHidden);

        ParameterTypeInt colorHidden = new ParameterTypeInt(
                PARAMETER_COLOR_HIDDEN, "Hidden dimension for the color network (e.g., 64).", 1, 1024, false);
        colorHidden.registerDependencyCondition(trainCondition);
        parameterTypes.add(colorHidden);

        ParameterTypeInt log2HashmapSize = new ParameterTypeInt(
                PARAMETER_LOG2_HASHMAP_SIZE, "Log2 of the hash map size (e.g., 20).", 10, 30, false);
        log2HashmapSize.registerDependencyCondition(trainCondition);
        parameterTypes.add(log2HashmapSize);

        ParameterTypeInt maxResolution = new ParameterTypeInt(
                PARAMETER_MAX_RESOLUTION, "Maximum resolution for encoding (e.g., 4096).", 128, 8192, false);
        maxResolution.registerDependencyCondition(trainCondition);
        parameterTypes.add(maxResolution);

        ParameterTypeBoolean noBgNerf = new ParameterTypeBoolean(
                PARAMETER_NO_BG_NERF, "Disable the background NeRF model.", false, false);
        noBgNerf.registerDependencyCondition(trainCondition);
        parameterTypes.add(noBgNerf);

        ParameterTypeInt bgHidden = new ParameterTypeInt(
                PARAMETER_BG_HIDDEN, "Hidden dimension for the background network (e.g., 32).", 1, 1024, false);
        bgHidden.registerDependencyCondition(trainCondition);
        parameterTypes.add(bgHidden);

        // Scheduling & Loop
        ParameterTypeBoolean noScheduler = new ParameterTypeBoolean(
                PARAMETER_NO_SCHEDULER, "Disable the learning rate scheduler.", false, false);
        noScheduler.registerDependencyCondition(trainCondition);
        parameterTypes.add(noScheduler);

        ParameterTypeDouble decayFactor = new ParameterTypeDouble(
                PARAMETER_DECAY_FACTOR, "Factor for learning rate decay (e.g., 10.0).", 1.0, 100.0, false);
        decayFactor.registerDependencyCondition(trainCondition);
        parameterTypes.add(decayFactor);

        ParameterTypeBoolean useStoredArgs = new ParameterTypeBoolean(
                PARAMETER_USE_STORED_ARGS, "Use arguments stored in the checkpoint.", false, false);
        useStoredArgs.registerDependencyCondition(trainCondition);
        parameterTypes.add(useStoredArgs);

        ParameterTypeInt outerSteps = new ParameterTypeInt(
                PARAMETER_OUTER_STEPS, "Total number of outer training steps (e.g., 20000).", 1, Integer.MAX_VALUE, false);
        outerSteps.registerDependencyCondition(trainCondition);
        parameterTypes.add(outerSteps);

        ParameterTypeInt innerIter = new ParameterTypeInt(
                PARAMETER_INNER_ITER, "Number of inner iterations (e.g., 8).", 1, 100, false);
        innerIter.registerDependencyCondition(trainCondition);
        parameterTypes.add(innerIter);

        ParameterTypeDouble innerLr = new ParameterTypeDouble(
                PARAMETER_INNER_LR, "Learning rate for inner iterations (e.g., 0.015).", 0.0, 1.0, false);
        innerLr.registerDependencyCondition(trainCondition);
        parameterTypes.add(innerLr);

        return parameterTypes;
    }


    private String generateJSON() throws UndefinedParameterError, JsonProcessingException {
        // Create the ObjectMapper
        ObjectMapper mapper = new ObjectMapper();

        ObjectNode rootNode = mapper.createObjectNode();
        String opMode = getParameterAsString(PARAMETER_OPERATION_MODE);
        // --------------------------------------------------------------------
        // 0. CORE FIELDS (Common to all 3: train, eval, view)
        // --------------------------------------------------------------------
        // Note: 'op' is handled by the method argument in this version.
        rootNode.put("op", opMode);
        rootNode.put(PARAMETER_SEED, getParameterAsInt(PARAMETER_SEED));
        rootNode.put(PARAMETER_DATASET, getParameterAsString(PARAMETER_DATASET));
        rootNode.put("data_path", getParameterAsString(PARAMETER_DATASET_PATH));
        rootNode.put("data_dirname", getParameterAsString(PARAMETER_DATA_DIRNAME));
        rootNode.put("mask_dirname", getParameterAsString(PARAMETER_MASK_DIRNAME));
        rootNode.put(PARAMETER_DOWNSCALE, getParameterAsDouble(PARAMETER_DOWNSCALE));
        rootNode.put("use_amp", getParameterAsBoolean(PARAMETER_USE_AMP));

        // Checkpoint path is common, but has different values/context
        rootNode.put(PARAMETER_CHECKPOINT_PATH, getParameterAsString(PARAMETER_CHECKPOINT_PATH));

        // ray_samples is common, but value changes for 'view'
        rootNode.put(PARAMETER_RAY_SAMPLES, getParameterAsInt(PARAMETER_RAY_SAMPLES));


        // --------------------------------------------------------------------
        // 1. EVAL and TRAIN COMMON FIELDS
        // --------------------------------------------------------------------
        if ("train".equals(opMode) || "eval".equals(opMode)) {
            // Rendering/Boundary
            rootNode.put(PARAMETER_NEAR, getParameterAsDouble(PARAMETER_NEAR));
            rootNode.put(PARAMETER_FAR, getParameterAsDouble(PARAMETER_FAR));
            // ray_samples already added above
            rootNode.put(PARAMETER_CHUNK_POINTS, getParameterAsInt(PARAMETER_CHUNK_POINTS));
            rootNode.put(PARAMETER_BOUNDARY_MARGIN, getParameterAsDouble(PARAMETER_BOUNDARY_MARGIN));

            // Batching & Data
            rootNode.put(PARAMETER_TEST_BATCH_SIZE, getParameterAsInt(PARAMETER_TEST_BATCH_SIZE));
            rootNode.put(PARAMETER_SUPPORT_RAYS, getParameterAsInt(PARAMETER_SUPPORT_RAYS));

            // Optimization
            rootNode.put(PARAMETER_OPTIMIZER, getParameterAsString(PARAMETER_OPTIMIZER));
            rootNode.put(PARAMETER_LR, getParameterAsDouble(PARAMETER_LR));
            rootNode.put(PARAMETER_ENCODING_LR, getParameterAsDouble(PARAMETER_ENCODING_LR));
            rootNode.put(PARAMETER_SIGMA_LR, getParameterAsDouble(PARAMETER_SIGMA_LR));
            rootNode.put(PARAMETER_COLOR_LR, getParameterAsDouble(PARAMETER_COLOR_LR));
            rootNode.put(PARAMETER_BG_LR, getParameterAsDouble(PARAMETER_BG_LR));

            // Checkpoint prefix
            rootNode.put(PARAMETER_PREFIX, getParameterAsString(PARAMETER_PREFIX));
        }


        // --------------------------------------------------------------------
        // 2. OPERATION-SPECIFIC FIELDS
        // --------------------------------------------------------------------

        if ("eval".equals(opMode)) {
            // Eval Unique
            rootNode.put(PARAMETER_TTO, getParameterAsString(PARAMETER_TTO));

        } else if ("train".equals(opMode)) {
            // Train Unique
            rootNode.put(PARAMETER_SAVE_STEP, getParameterAsInt(PARAMETER_SAVE_STEP));

            // Training Batch
            rootNode.put(PARAMETER_BATCH_SIZE, getParameterAsInt(PARAMETER_BATCH_SIZE));
            rootNode.put(PARAMETER_QUERY_RAYS, getParameterAsInt(PARAMETER_QUERY_RAYS));
            rootNode.put(PARAMETER_CELL_DIM, getParameterAsInt(PARAMETER_CELL_DIM));

            // Model Architecture
            rootNode.put(PARAMETER_NUM_SUBMODULES, getParameterAsInt(PARAMETER_NUM_SUBMODULES));
            rootNode.put(PARAMETER_SIGMA_DEPTH, getParameterAsInt(PARAMETER_SIGMA_DEPTH));
            rootNode.put(PARAMETER_COLOR_DEPTH, getParameterAsInt(PARAMETER_COLOR_DEPTH));
            rootNode.put(PARAMETER_DIM_HIDDEN, getParameterAsInt(PARAMETER_DIM_HIDDEN));
            rootNode.put(PARAMETER_COLOR_HIDDEN, getParameterAsInt(PARAMETER_COLOR_HIDDEN));
            rootNode.put(PARAMETER_LOG2_HASHMAP_SIZE, getParameterAsInt(PARAMETER_LOG2_HASHMAP_SIZE));
            rootNode.put(PARAMETER_MAX_RESOLUTION, getParameterAsInt(PARAMETER_MAX_RESOLUTION));
            rootNode.put(PARAMETER_NO_BG_NERF, getParameterAsBoolean(PARAMETER_NO_BG_NERF));
            rootNode.put(PARAMETER_BG_HIDDEN, getParameterAsInt(PARAMETER_BG_HIDDEN));

            // Scheduling & Loop
            rootNode.put(PARAMETER_NO_SCHEDULER, getParameterAsBoolean(PARAMETER_NO_SCHEDULER));
            rootNode.put(PARAMETER_DECAY_FACTOR, getParameterAsDouble(PARAMETER_DECAY_FACTOR));
            rootNode.put(PARAMETER_USE_STORED_ARGS, getParameterAsBoolean(PARAMETER_USE_STORED_ARGS));
            rootNode.put(PARAMETER_OUTER_STEPS, getParameterAsInt(PARAMETER_OUTER_STEPS));
            rootNode.put("inner_iter", getParameterAsInt(PARAMETER_INNER_ITER));
            rootNode.put(PARAMETER_INNER_LR, getParameterAsDouble(PARAMETER_INNER_LR));

        } else if ("view".equals(opMode)) {
            // View Unique
            rootNode.put("prefix", getParameterAsString(PARAMETER_FNAME));

        }

        // Use pretty printing to match the structure of your examples
        try {
            return mapper.writerWithDefaultPrettyPrinter().writeValueAsString(rootNode);
        } catch (Exception e) {
            // Handle exception
            return "{}";
        }
    }

    @Override
    public Pair<StreamGraph, List<StreamDataContainer>> getStreamDataInputs() throws UserError {
        StreamGraph graph = ((StreamingNest) getExecutionUnit().getEnclosingOperator()).getGraph();
        return new Pair<>(graph, Collections.emptyList());
    }

    @Override
    public List<StreamProducer> addToGraph(StreamGraph graph, List<StreamDataContainer> streamDataInputs) {
        String jsonContent = null;
        try {
            jsonContent = generateJSON();
        } catch (UndefinedParameterError e) {
            throw new RuntimeException(e);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }

        // Create a new stream producer for the JSON data
        JsonDataProducer producer = new JsonDataProducer.Builder(graph)
                .withJsonData(jsonContent)
                .build();
        graph.registerSource(producer);
        return Collections.singletonList(producer);
    }

    @Override
    public void deliverStreamDataOutputs(StreamGraph graph, List<StreamProducer> streamProducers) throws UserError {
        StreamDataContainer outData = new StreamDataContainer(graph, streamProducers.get(0));
        jsonOutput.deliver(outData);
    }

    @Override
    public long getId() {
        return 0;
    }

    @Override
    public void accept(StreamGraphNodeVisitor visitor) {

    }

    @Override
    public void doWork() throws OperatorException {
//
//        Pair<StreamGraph, List<StreamDataContainer>> inputs = getStreamDataInputs();
//        StreamGraph graph = inputs.getFirst();
//        logProcessing(graph.getName());
//
//        List<StreamProducer> streamProducers = null;
//        streamProducers = addToGraph(graph, inputs.getSecond());
//        deliverStreamDataOutputs(graph, streamProducers);

        if (getExecutionUnit().getEnclosingOperator() instanceof StreamingNest) {
            // Streaming nest behavior: use the graph-based approach.
            Pair<StreamGraph, List<StreamDataContainer>> inputs = getStreamDataInputs();
            StreamGraph graph = inputs.getFirst();
            logProcessing(graph.getName());
            List<StreamProducer> streamProducers = addToGraph(graph, inputs.getSecond());
            deliverStreamDataOutputs(graph, streamProducers);
        } else {
            // Not inside a streaming nest: just output JSON directly.
            try {
                String jsonContent = generateJSON();
                ObjectMapper mapper = new ObjectMapper();
                JsonNode jsonNode = mapper.readTree(jsonContent);
                // Create a single attribute called "json"
                Attribute jsonAttr = AttributeFactory.createAttribute("json", Ontology.STRING);
                ExampleSetBuilder builder = ExampleSets.from(jsonAttr).withExpectedSize(1);

                double[] row = new double[1];
                row[0] = jsonAttr.getMapping().mapString(jsonContent);

                builder.addRow(row);

                jsonOutput.deliver(builder.build());
//                logger.info("Standalone JSON output: " + jsonContent);
//                // Deliver the JSON string directly on the output port.
//                // (Depending on your output port configuration, you might need to adjust the port type.)
//                TableBuilder builder = Builders.newTableBuilder(1);
//
//                // 3) Add a single nominal column for the JSON data
//                builder.addNominal("json_column", i -> jsonContent);
//
//                // 4) Build the table (in this case, no parallelism is necessary)
//                Table table = builder.build(BeltTools.getContext(this));
//
//                // 5) Wrap the table in an IOTable
//                IOTable ioTable = new IOTable(table);
//                jsonOutput.deliver(ioTable);

            } catch (Exception e) {
                throw new OperatorException(e.getMessage(), e);
            }
        }
    }
}
