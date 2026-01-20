package TapasExplTreeViewer.clustering;

import javax.swing.*;
import java.awt.*;

public class ClusterContent {
  /**
   * Numeric identifiers of all original objects; need to be set from outside
   */
  public int objIds[]=null;
  /**
   * For each of N original objects contains true if the object is cluster member.
   */
  public boolean member[]=null;
  /**
   * Index of the object that is the medoid of the cluster
   */
  public int medoidIdx=-1;
  /**
   * Upper cluster in the hierarchy
   */
  public ClusterContent parent=null;
  /**
   * Lower clusters in the hierarchy
   */
  public ClusterContent children[]=null;
  /**
   * Depth of the hierarchy below this cluster; 0 if no children.
   */
  public int hierDepth=0;
  
  public void initialize(int nObjects) {
    member=new boolean[nObjects];
    for (int i=0; i<nObjects; i++)
      member[i]=false;
  }
  
  public int getMemberCount() {
    if (member==null)
      return 0;
    int n=0;
    for (int i=0; i<member.length; i++)
      if (member[i])
        ++n;
    return n;
  }
  
  public double getDiameter (double distanceMatrix[][]) {
    double d=0;
    for (int i=0; i<member.length-2; i++)
      if (member[i])
        for (int j=i+1; j<member.length-1; j++)
          if (member[j])
            d = Math.max(d,distanceMatrix[i][j]);
    return d;
  }
  
  public double getMRadius (double distanceMatrix[][]) {
    double d=0;
    for (int i=0; i<member.length-1; i++)
      if (i!=medoidIdx && member[i]) d = Math.max(d,distanceMatrix[medoidIdx][i]);
    return d;
  }
  
  public int getNClustersAtLevel(int level) {
    if (level>hierDepth)
      return getNClustersAtLevel(hierDepth);
    if (level==0)
      return 1;
    if (children==null)
      return 0;
    return children[0].getNClustersAtLevel(level-1)+children[1].getNClustersAtLevel(level-1);
  }
  
  public ClusterContent[] getClustersAtLevel(int level) {
    if (level>hierDepth)
      return getClustersAtLevel(hierDepth);
    if (level==0) {
      ClusterContent result[]={this};
      return result;
    }
    if (level==1)
      return children;
    if (children==null)
      return null;
    ClusterContent sub1[]=children[0].getClustersAtLevel(level-1),
        sub2[]=children[1].getClustersAtLevel(level-1);
    if (sub1==null)
      return sub2;
    if (sub2==null)
      return sub1;
    ClusterContent result[]=new ClusterContent[sub1.length+sub2.length];
    for (int i=0; i<sub1.length; i++)
      result[i]=sub1[i];
    for (int i=0; i<sub2.length; i++)
      result[sub1.length+i]=sub2[i];
    return result;
  }
  
  public void setObjIds(int[] objIds) {
    this.objIds = objIds;
  }
  
  public void drawHierarchy(Graphics g, int x0, int y0, int stepX, int stepY, int width, int height) {
    if (g==null || getMemberCount()<1)
      return;
    if (stepX<=0)
      stepX=width/(hierDepth+1);
    if (stepY<=0)
      stepY=height/member.length;
    int n=getMemberCount();
    int oId=(objIds!=null)?objIds[medoidIdx]:medoidIdx;
    String txt=(n>1)?"("+n+";"+Integer.toString(oId)+")":Integer.toString(oId);
    g.drawString(txt,x0,y0+stepY-g.getFontMetrics().getDescent()-1);
    if (children!=null) {
      int h0=stepY*children[0].getMemberCount();
      int y1=y0+stepY-1, y2=y1+h0;
      g.drawLine(x0,y1,x0+stepX,y1);
      g.drawLine(x0+stepX/2,y1,x0+stepX/2,y2);
      g.drawLine(x0+stepX/2,y2,x0+stepX,y2);
      children[0].drawHierarchy(g,x0+stepX,y0,stepX,stepY,width,height);
      children[1].drawHierarchy(g,x0+stepX,y0+h0,stepX,stepY,width,height);
    }
  }
  
  public JPanel makePanel() {
    if (getMemberCount()<1)
      return null;
    JPanel p=new JPanel() {
      public Dimension getPreferredSize() {
        return new Dimension(50*(hierDepth+1),20*getMemberCount());
      }
      public void paintComponent(Graphics g) {
        g.setColor(getBackground());
        g.fillRect(0,0,getWidth()+1,getHeight()+1);
        g.setColor(Color.black);
        drawHierarchy(g,1,1,0,0,getWidth()-2,getHeight()-2);
      }
    };
    return p;
  }
  
  public static ClusterContent joinClusters(ClusterContent cc1, ClusterContent cc2) {
    if (cc1==null || cc2==null || cc1.getMemberCount()<1 || cc2.getMemberCount()<1)
      return null;
    ClusterContent cc=new ClusterContent();
    cc.member=new boolean[cc1.member.length];
    for (int i=0; i<cc.member.length; i++)
      cc.member[i]= cc1.member[i] || cc2.member[i];
    cc.hierDepth=1+Math.max(cc1.hierDepth,cc2.hierDepth);
    cc.children=new ClusterContent[2];
    cc.children[0]=cc1;
    cc.children[1]=cc2;
    cc1.parent=cc;
    cc2.parent=cc;
    return cc;
  }
  
  public static double distanceBetweenClusters(ClusterContent cc1, ClusterContent cc2, double distances[][]) {
    if (cc1==null || cc2==null || cc1.getMemberCount()<1 || cc2.getMemberCount()<1 || distances==null)
      return Double.NaN;
    int nDistances=0;
    double sumDistances=0;
    for (int i=0; i<cc1.member.length; i++)
      if (cc1.member[i])
        for (int j=0; j<cc2.member.length; j++)
          if (cc2.member[j] && !Double.isNaN(distances[i][j])) {
            ++nDistances;
            sumDistances+=distances[i][j];
          }
    if (nDistances<1)
      return Double.NaN;
    return sumDistances/nDistances;
  }
}
