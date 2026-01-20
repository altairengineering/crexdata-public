package TapasExplTreeViewer.clustering;

import it.unipi.di.sax.optics.AnotherOptics;
import it.unipi.di.sax.optics.ClusterListener;
import it.unipi.di.sax.optics.ClusterObject;
import it.unipi.di.sax.optics.DistanceMeter;

import javax.swing.*;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.TreeSet;

public class OPTICS_Runner
    implements DistanceMeter<ClusterObject<Integer>>, ClusterListener {
  /**
   * Matrix of distances between the objects to cluster
   */
  public double distances[][]=null;
  /**
   * The objects prepared for clustering
   */
  protected ArrayList<ClusterObject<Integer>> objToCluster=null;
  /**
   * OPTICS parameter: the maximal distance (neighbourhood radius)
   */
  public double neibRadius=Double.NaN;
  /**
   * OPTICS parameter: the minimal number of neighbours of a core object
   */
  public int minNeighbors=5;
  /**
   * Indicates which objects have been already ordered
   */
  protected boolean ordered[]=null;
  /**
   * The objects ordered by the clustering algorithm
   */
  protected ArrayList<ClusterObject> objOrdered=null;
  
  public void setDistanceMatrix(double distances[][]) {
    this.distances = distances;
    if (distances!=null) {
      objToCluster=new ArrayList<ClusterObject<Integer>>(distances.length);
      for (int i=0; i<distances.length; i++) {
        ClusterObject<Integer> clObj=new ClusterObject<Integer>(new Integer(i));
        objToCluster.add(clObj);
      }
    }
  }
  /**
   * Tries to select clustering parameters based on statistics of distances
   */
  public void doClustering() {
    TreeSet<Double> distSet=new TreeSet<Double>();
    for (int i=0; i<distances.length-1; i++)
      for (int j=i+1; j<distances.length; j++)
        distSet.add(distances[i][j]);
    ArrayList<Double> distList=new ArrayList<Double>(distSet);
    int idx=Math.round(0.025f*distList.size());
    if (idx<5) {
      idx = Math.round(0.125f * distList.size());
      minNeighbors=3;
    }
    if (idx<5) {
      idx = Math.round(0.25f * distList.size());
      minNeighbors=2;
    }
    doClustering(distList.get(idx),minNeighbors);
  }
  
  public void doClustering(double neibRadius, int minNeighbors) {
    if (objToCluster==null || objToCluster.size()<=minNeighbors)
      return;
    this.neibRadius=neibRadius;
    this.minNeighbors=minNeighbors;
    System.out.println("Starting OPTICS clustering; max distance = "+neibRadius+
                           "; min N neighbors = "+minNeighbors);
    objOrdered=null;
    ordered=null;
    AnotherOptics optics = new AnotherOptics(this);
    optics.addClusterListener(this);
    SwingWorker worker=new SwingWorker() {
      @Override
      public Boolean doInBackground(){
        optics.optics(objToCluster,neibRadius,minNeighbors);
        return true;
      }
      @Override
      protected void done() {
        clusteringDone();
      }
    };
    worker.execute();
  }
  
  public double getNeibRadius() {
    return neibRadius;
  }
  
  public int getMinNeighbors() {
    return minNeighbors;
  }
  
  public double distance(ClusterObject<Integer> o1, ClusterObject<Integer> o2) {
    if (o1==null || o2==null) return Double.POSITIVE_INFINITY;
    if (distances==null) return Double.POSITIVE_INFINITY;
    return distances[o1.getOriginalObject()][o2.getOriginalObject()];
  }
  
  public Collection<ClusterObject<Integer>> neighbors(ClusterObject<Integer> core,
                                                      Collection<ClusterObject<Integer>> objects,
                                                      double epsilon){
    TreeSet<ObjectWithMeasure> neighborsWithDistances=new TreeSet<ObjectWithMeasure>();
    for(Iterator<ClusterObject<Integer>> i = objects.iterator(); i.hasNext(); ){
      ClusterObject<Integer> o = i.next();
      if(!o.equals(core)){
        double dist = distance(o, core);
        if (dist <=epsilon){
          ObjectWithMeasure om=new ObjectWithMeasure(o,dist);
          neighborsWithDistances.add(om);
        }
      }
    }
    ArrayList<ClusterObject<Integer>> neiObj=
        new ArrayList<ClusterObject<Integer>>(Math.max(1,neighborsWithDistances.size()));
    for (Iterator<ObjectWithMeasure> it=neighborsWithDistances.iterator(); it.hasNext();) {
      ObjectWithMeasure om=it.next();
      neiObj.add((ClusterObject<Integer>)om.obj);
    }
    return neiObj;
  }
  
  /**
   * Receives an object from the clustering tool
   */
  public void emit(ClusterObject o) {
    if (ordered==null) {
      ordered=new boolean[objToCluster.size()];
      for (int i=0; i<ordered.length; i++)
        ordered[i]=false;
    }
    int idx=-1;
    if (o.getOriginalObject() instanceof ClusterObject) {
      ClusterObject origObj=(ClusterObject)o.getOriginalObject();
      if (origObj.getOriginalObject() instanceof Integer) {
        idx = (Integer) origObj.getOriginalObject();
        if (ordered[idx])
          return;
      }
    }
    if (objOrdered==null)
      objOrdered=new ArrayList<ClusterObject>(objToCluster.size());
    objOrdered.add(o);
    ordered[idx]=true;
    if (objOrdered.size()%250==0)
      System.out.println("OPTICS clustering: "+objOrdered.size()+" objects put in order");
  }
  
  
  protected void clusteringDone() {
    if (objOrdered == null || objOrdered.isEmpty()) {
      System.out.println("OPTICS clustering failed!");
      return;
    }
    System.out.println("OPTICS clustering finished!");
  }
  
  
  public static ClustersAssignments makeClusters(ArrayList<ClusterObject> objOrdered,
                                                 double distThreshold) {
    if (objOrdered==null || objOrdered.isEmpty())
      return null;
    ClustersAssignments clAss=new ClustersAssignments();
    clAss.clusters=new int[objOrdered.size()];
    clAss.objIndexes=new int[objOrdered.size()];
    clAss.nClusters=0;
    int currClusterSize=0;
    for (int i=0; i<objOrdered.size(); i++) {
      clAss.clusters[i]=-1; //noise
      clAss.objIndexes[i]=-1;
      ClusterObject clObj=objOrdered.get(i);
      if (clObj.getOriginalObject() instanceof ClusterObject) {
        ClusterObject origObj=(ClusterObject)clObj.getOriginalObject();
        if (origObj.getOriginalObject() instanceof Integer)
          clAss.objIndexes[i]=(Integer)origObj.getOriginalObject();
      }
      if (!Double.isNaN(distThreshold)) {
        double d = clObj.getReachabilityDistance();
        boolean reachable=!Double.isNaN(d) && !Double.isInfinite(d) && d < distThreshold;
        boolean nextReachable=false;
        if (!reachable && i+1<objOrdered.size()) {
          d=objOrdered.get(i+1).getReachabilityDistance();
          nextReachable=!Double.isNaN(d) && !Double.isInfinite(d) && d < distThreshold;
        }
        if (reachable || nextReachable) {
          if (clAss.nClusters == 0 || nextReachable || (i > 0 && (clAss.clusters[i - 1] < 0))) { //new cluster
            if (currClusterSize > 0) {
              if (clAss.minSize < 1 || clAss.minSize > currClusterSize)
                clAss.minSize = currClusterSize;
              if (clAss.maxSize < currClusterSize)
                clAss.maxSize = currClusterSize;
            }
            clAss.clusters[i] = clAss.nClusters++;
            currClusterSize = 1;
          }
          else {
            clAss.clusters[i] = clAss.clusters[i - 1]; //continuation of the previous cluster
            ++currClusterSize;
          }
        }
        else {
          ++clAss.nNoise;
        }
      }
    }
    return clAss;
  }
}
