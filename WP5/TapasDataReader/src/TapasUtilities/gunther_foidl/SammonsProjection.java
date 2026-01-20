package TapasUtilities.gunther_foidl;

import java.util.Random;

/**
 * Created by IntelliJ IDEA.
 * User: Natascha
 * Date: Nov 4, 2009
 * Time: 5:59:39 PM
 * The original code was written by Günther M. Foidl;
 * see http://www.codeproject.com/KB/recipes/SammonProjection.aspx.
 * Implements the algorithm of Sammon's projection.
 * Sammon's projection is a nonlinear projection method to map a
 * high dimensional space onto a space of lower dimensionality.
 *
 * Usage:
 * SammonsProjection projection = new SammonsProjection(_inputData,2,1000);
 * projection.CreateMapping();
 */
public class SammonsProjection {
  protected int _maxIteration;
  private double _lambda = 1;		// Startwert
  private int[] _indicesI;
  private int[] _indicesJ;
  protected double[][] _distanceMatrix;

  public double[][] getDistanceMatrix() {
    return _distanceMatrix;
  }

  public void setDistanceMatrix(double[][] _distanceMatrix) {
    this._distanceMatrix = _distanceMatrix;
  }

  // The number of input vectors
  public int Count=0;

  // The dimension in that the projection should be performed.
  public int OutputDimension=2;
  public int getOutputDimension() {
    return OutputDimension;
  }
  public void setOutputDimension(int outputDimension) {
    OutputDimension = outputDimension;
  }
  /**
   * The projected vector. The length of the first dimension equals Count
   * (the number of the input vectors), the length of the second
   * dimension is OutputDimension, e.g. 2. 
   */
  public double[][] Projection=null;
  public double[][] getProjection() {
    return Projection;
  }
  public void setProjection(double proj[][]) {
    Projection=proj;
  }
  public boolean hasProjection() {
    return Projection!=null && Projection.length>0;
  }
  // The number of iterations.
  public int Iteration;
  public int getIteration() {
    return Iteration;
  }
  public void setIteration(int iteration) {
    Iteration = iteration;
  }
  /**
   * Whether to use Euclidean or Manhattan distance
   */
  public boolean useEuclid =true;
  public boolean doesUseEuclid() {
    return useEuclid;
  }
  public void setUseEuclid(boolean useEuclid) {
    this.useEuclid = useEuclid;
  }

  /**
   * Creates a new instance of Sammon's Mapping with the default number of iterations
   * @param inputData - The input-vectors
   * @param outputDimension - The dimension of the projection.
   */
  public SammonsProjection(double[][] inputData, int outputDimension, boolean useEuclid) {
    this(inputData,outputDimension,inputData.length * (int)1e4,useEuclid);
  }
  /**
   * Creates a new instance of Sammon's Mapping.
   * @param inputData - The input-vectors
   * @param outputDimension - The dimension of the projection.
   * @param maxIteration - Maximum number of iterations. For a statistical acceptable accuracy
	 * this should be 10e4...1e5 times the number of points. It has shown
	 * that a few iterations (100) yield a good projection.
   */
  public SammonsProjection(double[][] inputData, int outputDimension,
                           int maxIteration, boolean useEuclid){
    if (inputData == null || inputData.length == 0)
      return;
    //-----------------------------------------------------------------
    this.Count=inputData.length;
    this.OutputDimension = outputDimension;
    _maxIteration = maxIteration;
    this.useEuclid =useEuclid;

    // Initialize the projection:
    Initialize(inputData);

    // Create the indices-arrays:
    _indicesI = new int[this.Count];
    _indicesJ = new int[this.Count];
    for (int i=0; i<this.Count; i++)
      _indicesI[i]=_indicesJ[i]=i;
	}
  /**
   * Creates a new instance of Sammon's mapping given a pre-computed distance
   * matrix and an initial projection (e.g. a result of SOM).
   * initialProjection may be null
   * outputDimension is ignored when initialProjection is not null
   */
  public SammonsProjection(double[][] distanceMatrix,
                           double[][] initialProjection,
                           int outputDimension,
                           int maxIterations,
                           boolean useEuclid) {
    if (distanceMatrix==null)
      return;
    _distanceMatrix=distanceMatrix;
    this.Count=_distanceMatrix.length;
    if (initialProjection!=null) {
      this.Projection=initialProjection;
      this.OutputDimension=initialProjection[0].length;

      //the following code is intended for avoiding large displacements
      //with respect to the initial projection
      this.Iteration = maxIterations;
      _maxIteration = this.Iteration+ maxIterations;
      double ratio = (double)this.Iteration / _maxIteration;
      _lambda = Math.pow(0.01, ratio);
    }
    else {
      this.OutputDimension = outputDimension;
      _maxIteration = maxIterations;
      _lambda = 1;		// Startwert
      createRandomProjection();
    }

    this.useEuclid =useEuclid;
    // Create the indices-arrays:
    _indicesI = new int[this.Count];
    _indicesJ = new int[this.Count];
    for (int i=0; i<this.Count; i++)
      _indicesI[i]=_indicesJ[i]=i;
  }

  protected void createRandomProjection () {
    // Initialize random points for the projection:
    double maxD=0;
    for (int i=0; i<_distanceMatrix.length; i++)
      for (int j=i+1; j<_distanceMatrix[i].length; j++)
        if (!Double.isNaN(_distanceMatrix[i][j]) && _distanceMatrix[i][j]>maxD)
          maxD=_distanceMatrix[i][j];
    Random rnd = new Random();
    Projection = new double[Count][OutputDimension];
    for (int i = 0; i < Count; i++)
      for (int j = 0; j < OutputDimension; j++)
        Projection[i][j] = rnd.nextDouble() * maxD; //initially it was Count
  }

  public void reset (int maxIterations, boolean keepPreviousProjection) {
    this.Iteration = 0;
    _maxIteration = maxIterations;
    _lambda = 1;		// Startwert

    if (!keepPreviousProjection)
      createRandomProjection();
  }

  /**
   * Initializes the algorithm.
   */
  private void Initialize(double[][] inputData) {
    Count=inputData.length;
    if (_distanceMatrix==null)
      _distanceMatrix = CalculateDistanceMatrix(Count,inputData,useEuclid);

    createRandomProjection();
  }

  /**
   * Runs all the iterations and thus create the mapping.
   */
  public void CreateMapping() {
    int i0=this.Iteration;
    for (int i = _maxIteration; i >= i0; i--)
      this.Iterate();
  }

  /**
   * Runs the given number of iterations
   */
  public void runMore (int nIterations) {
    //the following code is intended for avoiding large displacements
    //with respect to the initial projection
/*
    this.Iteration = nIterations;
    _maxIteration = this.Iteration+ nIterations;
*/
    _maxIteration+=nIterations;
    double ratio = (double)this.Iteration / _maxIteration;
    _lambda = Math.pow(0.01, ratio);

    for (int i=0; i<nIterations; i++)
      this.Iterate();
  }
  /**
   * Computes the Kruskal's stress (formula 1)
   */
  public double computeStress () {
    if (_distanceMatrix==null || Projection==null)
      return Double.NaN;
/*
    double maxd0=0, maxD=0;
    for (int i=0; i<_distanceMatrix.length; i++)
      for (int j=i+1; j<_distanceMatrix.length; j++) {
        double Dij =
          (useEuclid)?getEuclideanDistance(Projection[i],Projection[j]):
                      getManhattanDistance(Projection[i],Projection[j]);
        if (Double.isNaN(Dij))
          continue;
        if (Dij>maxD) maxD=Dij;
        double dij = _distanceMatrix[i][j];
        if (dij>maxd0) maxd0=dij;
      }
    if (maxd0==0 || maxD==0)
      return Double.NaN;
*/
    double sumDiff=0, sumDist=0;
    for (int i=0; i<_distanceMatrix.length; i++)
      for (int j=i+1; j<_distanceMatrix.length; j++) {
        double dij = _distanceMatrix[i][j];
        if (Double.isNaN(dij))
          continue;
        double Dij =
          (useEuclid)?getEuclideanDistance(Projection[i],Projection[j]):
                      getManhattanDistance(Projection[i],Projection[j]);
        if (Double.isNaN(Dij))
          continue;
        //Dij/=maxD;
        //dij/=maxd0;
        sumDist+=dij*dij;
        double diff=Dij-dij;
        sumDiff+=diff*diff;
      }
    if (sumDist==0)
      return Double.NaN;
    return Math.sqrt(sumDiff/sumDist);
  }

  /**
   * Performs one iteration of the (heuristic) algorithm.
   */
  public void Iterate() {
    // Shuffle the indices-array for random pick of the points:
    FisherYatesShuffle(_indicesI);
    FisherYatesShuffle(_indicesJ);

    for (int i = 0; i < _indicesI.length; i++)
      for (int j = 0; j < _indicesJ.length; j++) {
        int ii=_indicesI[i], jj=_indicesJ[j];
        if (ii == jj)
          continue;
        double dij = _distanceMatrix[ii][jj];
        if (Double.isNaN(dij)) {
          //System.out.println("NaN distance between "+ii+" and "+jj);
          continue;
        }
        double Dij =
          (useEuclid)?getEuclideanDistance(Projection[ii],Projection[jj]):
                      getManhattanDistance(Projection[ii],Projection[jj]);
        if (Double.isNaN(Dij)) {
          //System.out.println("NaN distance in the projection between "+ii+" and "+jj);
          continue;
        }
        // Avoid division by zero:
        if (Dij == 0)
          Dij = 1e-10;
        double delta = _lambda * (dij - Dij) / Dij;

        for (int k = 0; k < Projection[ii].length; k++) {
          double correction =
            delta * (Projection[ii][k] - Projection[jj][k]);
          if (Double.isNaN(correction)) {
            //System.out.println("NaN correction for the positions "+ii+" and "+jj);
            break;
          }

          Projection[ii][k] += correction;
          Projection[jj][k] -= correction;
        }
    }
    // Reduce lambda monotonically:
    ReduceLambda();
  }

  private void ReduceLambda() {
    this.Iteration++;

    // Herleitung über den Ansatz y(t) = k.exp(-l.t).
    double ratio = (double)this.Iteration / _maxIteration;

    // Start := 1, Ende := 0.01
    _lambda = Math.pow(0.01, ratio);
  }

  public static double getManhattanDistance (double[] vec1, double[] vec2) {
    if (vec1==null || vec2==null || vec1.length!=vec2.length)
      return 0;
    double distance = 0;
    int count=0;
    for (int i = 0; i < vec1.length; i++)
      if (!Double.isNaN(vec1[i]) && !Double.isNaN(vec2[i])) {
        distance += Math.abs(vec1[i] - vec2[i]);
        ++count;
      }
    if (count>0)
      return distance;
    return Double.NaN;
  }

  public static double getEuclideanDistance( double v1[], double v2[] ) {
    if (v1==null || v2==null || v1.length!=v2.length)
      return 0;
    double squaressum=0;
    int count=0;
    for ( int i=0; i<v1.length; i++ )
      if (!Double.isNaN(v1[i]) && !Double.isNaN(v2[i])){
        double d=v1[i]-v2[i];
        squaressum += d*d;
        ++count;
      }
    if (count<1)
      return Double.NaN;
    double distance=Math.sqrt(squaressum);
    return distance;
  }
  /**
   * In order to pick the vectors randomly, I quite often use an index-array
   * that is shuffled. The shuffling is done by the Fisher-Yates-Shuffle algorithm.
   */
  public static void FisherYatesShuffle(int array[]) {
    Random rnd=new Random();
    for (int i = array.length - 1; i > 0; i--) {
      // Pick random position:
      int pos = Math.round(rnd.nextFloat()*(array.length-1));
      if (pos!=i) {
        // Swap:
        int tmp = array[i];
        array[i] = array[pos];
        array[pos] = tmp;
      }
    }
  }

  /**
   * @param nVectors the number of data items (vectors)
   * @param inputData - the data item (vectors); the length of the first dimension is nvectors
   * @return distance matrix with the size nVectors x nVectors
   */
  public static double[][] CalculateDistanceMatrix(int nVectors, double[][] inputData, boolean useEuclid) {
    double[][] distanceMatrix = new double[nVectors][nVectors];
    for (int i = 0; i < nVectors; i++) {
      distanceMatrix[i][i] = Double.NaN;
      for (int j = i+1; j < nVectors; j++)
        distanceMatrix[i][j] = distanceMatrix[j][i] =
          (useEuclid)?getEuclideanDistance(inputData[i],inputData[j]):
                      getManhattanDistance(inputData[i],inputData[j]);
    }
    return distanceMatrix;
  }
}
