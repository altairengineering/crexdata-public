package TapasExplTreeViewer.clustering;

import java.util.ArrayList;

public class HierarchicalClusterer {
  /**
   * @param distances - Matrix of distances between the objects to cluster
   * @return top cluster in the hierarchy or null if failed
   */
  public static ClusterContent doClustering(double distances[][],
                                            boolean useDistancesBetweenMedoids) {
    if (distances==null || distances.length<3)
      return null; //nothing to cluster
    
    int nObjects=distances.length;
    
    double sumDistances[]=new double[nObjects];
    for (int i=0; i<nObjects; i++) {
      if (distances[i]==null)
        return null; //no distances for an object
      sumDistances[i]=0;
      for (int j=0; j<distances[i].length; j++)
        if (!Double.isNaN(distances[i][j]))
          sumDistances[i]+=distances[i][j];
      if (sumDistances[i]==0)
        return null; //no distances for an object
    }
  
    ArrayList<ClusterContent> clusters=new ArrayList<ClusterContent>(nObjects*(nObjects-1));
    int clusterIdx[]=new int[nObjects];
    boolean isMedoid[]=new boolean[nObjects];
    for (int i=0; i<nObjects; i++) {
      ClusterContent cc=new ClusterContent();
      cc.initialize(nObjects);
      cc.member[i]=true;
      cc.medoidIdx=i;
      clusterIdx[i]=clusters.size();
      isMedoid[i]=true;
      clusters.add(cc);
    }
    
    do {
      int cIdx1=-1, cIdx2=-1;
      if (useDistancesBetweenMedoids) {
        //distance between clusters ::= distance between their medoids
        int idx1 = -1, idx2 = -1;
        double minD = Double.NaN;
        for (int i = 0; i < nObjects - 1; i++)
          if (isMedoid[i])
            for (int j = i + 1; j < nObjects; j++)
              if (isMedoid[j]) {
                if (Double.isNaN(minD) || (!Double.isNaN(distances[i][j]) && distances[i][j] < minD)) {
                  minD = distances[i][j];
                  idx1 = i;
                  idx2 = j;
                }
              }
        if (Double.isNaN(minD))
          break;
        cIdx1 = clusterIdx[idx1];
        cIdx2 = clusterIdx[idx2];
      }
      else {
        //distance between clusters ::= mean pairwise distance between members
        double minD = Double.NaN;
        for (int i = 0; i < clusters.size() - 1; i++)
          if (clusters.get(i).parent == null) //not included in another cluster
            for (int j = i + 1; j < clusters.size(); j++)
              if (clusters.get(j).parent == null) {//not included in another cluster
                Double d = ClusterContent.distanceBetweenClusters(clusters.get(i), clusters.get(j), distances);
                if (!Double.isNaN(d) && (Double.isNaN(minD) || minD > d)) {
                  minD = d;
                  cIdx1 = i;
                  cIdx2 = j;
                }
              }
        if (Double.isNaN(minD))
          break;
      }
      
      //join two closest clusters
      ClusterContent cc1=clusters.get(cIdx1), cc2=clusters.get(cIdx2),
          cc=ClusterContent.joinClusters(cc1,cc2);
      
      //determine the medoid of the joint cluster
      if (cc.getMemberCount()==2) {
        //select the member with the smallest sum of distances to all other objects
        cc.medoidIdx=(sumDistances[cIdx1]<sumDistances[cIdx2])?cIdx1:cIdx2;
      }
      else {
        double minDist=Double.NaN;
        for (int i=0; i<nObjects; i++)
          if (cc.member[i]) {
            double dSum=0;
            for (int j=0; j<nObjects; j++)
              if (cc.member[j] && !Double.isNaN(distances[i][j]))
                dSum+=distances[i][j];
            if (Double.isNaN(minDist) || minDist>dSum)  {
              minDist=dSum;
              cc.medoidIdx=i;
            }
          }
      }
      for (int i=0; i<nObjects; i++)
        if (cc.member[i]) {
          clusterIdx[i]=clusters.size();
          isMedoid[i]=cc.medoidIdx==i;
        }
      clusters.add(cc);
      if ((clusters.size()-nObjects)%50==0)
        System.out.println(Integer.toString(clusters.size()-nObjects)+" clusters have been created");
    } while (clusters.get(clusters.size()-1).getMemberCount()<nObjects);
    
    return clusters.get(clusters.size()-1);
  }
}
