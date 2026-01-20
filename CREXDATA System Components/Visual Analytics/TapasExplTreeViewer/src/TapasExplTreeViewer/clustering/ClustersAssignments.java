package TapasExplTreeViewer.clustering;

import it.unipi.di.sax.optics.ClusterObject;

import java.util.ArrayList;

public class ClustersAssignments {
  /**
   * Objects ordered by the clustering algorithm
   */
  public ArrayList<ClusterObject<Integer>> objOrdered = null;
  /**
   * Indexes of the clustered objects in the original list of the objects to cluster
   */
  public int objIndexes[]=null;
  /**
   * Cluster numbers assigned to the objects, starting from 0; -1 means noise
   */
  public int clusters[]=null;
  /**
   * Number of clusters
   */
  public int nClusters=0;
  /**
   * Number of objects in "noise"
   */
  public int nNoise=0;
  /**
   * Smallest and largest cluster sizes
   */
  public int minSize=0, maxSize=0;
}
