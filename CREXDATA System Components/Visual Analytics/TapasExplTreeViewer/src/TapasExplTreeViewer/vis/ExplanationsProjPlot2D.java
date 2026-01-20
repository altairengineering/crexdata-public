package TapasExplTreeViewer.vis;

import TapasDataReader.CommonExplanation;
import TapasExplTreeViewer.MST.Edge;
import TapasExplTreeViewer.MST.Vertex;
import TapasExplTreeViewer.ui.ShowSingleRule;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;

public class ExplanationsProjPlot2D extends ProjectionPlot2D {
  public static int minDotRadius=4, maxDotRadius=20;
  public static float hsbRed[]=Color.RGBtoHSB(255,0,0,null);
  public static float hsbBlue[]=Color.RGBtoHSB(0,0,255,null);
  public static Color linkColor=new Color(90,90,90,60);
  
  public ArrayList<CommonExplanation> explanations = null;
  public ArrayList<CommonExplanation> fullSet = null;
  public Hashtable<String,float[]> attrMinMax=null;
  public Vector<String> attrs=null;
  public Vector<float[]> minmax=null;
  /**
   * Features that were used in computing the distances
   */
  public ArrayList<String> features=null;

  /**
   * The graphs represent connections between rules when they are aggregated.
   * The labels of the vertices are string representations of the rule indexes in the list.
   */
  public HashSet<ArrayList<Vertex>> graphs=null;
  /**
   * If the explanations (rules) are members of unions (which may be represented by the graphs),
   * this array specifies the integer identifiers (indexes) of the unions.
   */
  public int unionIds[]=null;
  
  public int maxNUses = 0;
  public int minAction=Integer.MAX_VALUE, maxAction=Integer.MIN_VALUE;
  public boolean sameAction=true;
  public double minQ=Double.NaN, maxQ=Double.NaN;
  public int maxRadius=maxDotRadius;

  public ExplanationsProjPlot2D(Hashtable<String,float[]> attrMinMax, Vector<String> attrs, Vector<float[]> minmax){
    this.attrMinMax=attrMinMax; this.attrs=attrs; this.minmax=minmax;
    ToolTipManager.sharedInstance().registerComponent(this);
    ToolTipManager.sharedInstance().setDismissDelay(Integer.MAX_VALUE);
  }
  
  public ExplanationsProjPlot2D(Hashtable<String,float[]> attrMinMax,
                                Vector<String> attrs, Vector<float[]> minmax,
                                ArrayList<CommonExplanation> explanations,
                                ArrayList<CommonExplanation> fullSet,
                                double coords[][]) {
    super(coords);
    this.attrMinMax=attrMinMax; this.attrs=attrs; this.minmax=minmax;
    ToolTipManager.sharedInstance().registerComponent(this);
    ToolTipManager.sharedInstance().setDismissDelay(Integer.MAX_VALUE);
    setExplanations(explanations,fullSet);
  }
  
  public void setExplanations(ArrayList<CommonExplanation> explanations,
                              ArrayList<CommonExplanation> fullSet) {
    this.explanations = explanations;
    maxNUses = 0;
    ArrayList<CommonExplanation> exSet=(fullSet==null)?explanations:fullSet;
    if (exSet != null)
      for (int i = 0; i < exSet.size(); i++) {
        CommonExplanation ex=exSet.get(i);
        if (maxNUses < ex.nUses)
          maxNUses = ex.nUses;
        if (minAction>ex.action)
          minAction=ex.action;
        if (maxAction<ex.action)
          maxAction=ex.action;
        if (!Double.isNaN(ex.meanQ)) {
          if (Double.isNaN(minQ) || minQ>ex.meanQ)
            minQ=ex.minQ;
          if (Double.isNaN(maxQ) || maxQ<ex.meanQ)
            maxQ=ex.meanQ;
        }
      sameAction=minAction==maxAction;
    }
    if (explanations != null && !explanations.equals(exSet)) {
      maxNUses=0;
      for (int i = 0; i < explanations.size(); i++) {
        CommonExplanation ex = explanations.get(i);
        if (maxNUses < ex.nUses)
          maxNUses = ex.nUses;
      }
    }
    maxRadius=(maxNUses<maxDotRadius)?minDotRadius+maxNUses-1:maxDotRadius;
    off_Valid=off_selected_Valid=false;
    if (isShowing())
      repaint();
  }

  /**
   * Features that were used in computing the distances
   */
  public ArrayList<String> getFeatures() {
    return features;
  }
  /**
   * Features that were used in computing the distances
   */
  public void setFeatures(ArrayList<String> features) {
    this.features = features;
  }

  public HashSet<ArrayList<Vertex>> getGraphs() {
    return graphs;
  }
  
  public void setGraphs(HashSet<ArrayList<Vertex>> graphs) {
    this.graphs = graphs;
  }
  
  public int[] getUnionIds() {
    return unionIds;
  }
  
  public void setUnionIds(int[] unionIds) {
    this.unionIds = unionIds;
  }
  
  public void drawPoint(Graphics2D g, int pIdx, int x, int y, boolean highlighted, boolean selected) {
    if (explanations == null || explanations.isEmpty() || pIdx < 0 || pIdx >= explanations.size()) {
      super.drawPoint(g, pIdx, x, y, highlighted, selected);
      return;
    }
    CommonExplanation ex=explanations.get(pIdx);
    int dotRadius=minDotRadius+Math.round(1f*ex.nUses/maxNUses*(maxRadius-minDotRadius)),
      dotDiameter=2*dotRadius;
    Color color=dotColor;
    if (!highlighted && !selected) {
      color=(sameAction)?getColorForQ(ex.meanQ,minQ,maxQ):getColorForAction(ex.action,minAction,maxAction);
      g.setColor(color);
      g.fillOval(x-dotRadius-1,y-dotRadius-1,dotDiameter+2,dotDiameter+2);
    }
    else
      if (highlighted) {
        g.setColor(highlightFillColor);
        g.fillOval(x-dotRadius-1,y-dotRadius-1,dotDiameter+2,dotDiameter+2);
      }
    Stroke origStr=(selected || highlighted)?g.getStroke():null;
    if (selected || highlighted)
      g.setStroke(strokeSelected);
    g.setColor((highlighted)?highlightColor:(selected)?selectColor:color.darker());
    g.drawOval(x-dotRadius,y-dotRadius,dotDiameter,dotDiameter);
    if (origStr!=null)
      g.setStroke(origStr);
  }
  
  public void drawPoints(Graphics2D g) {
    super.drawPoints(g);
    drawLinks(g);
  }
  
  public void drawLinks(Graphics2D gr) {
    if (graphs==null || graphs.isEmpty())
      return;
    if (px==null || py==null)
      return;
    gr.setColor(linkColor);
    for (ArrayList<Vertex> graph:graphs)
      drawLinks(gr,graph);
  }
  
  public void drawLinks(Graphics2D gr, ArrayList<Vertex> graph) {
    if (graph==null || graph.isEmpty())
      return;
    for (Vertex v:graph) {
      int idx0=-1;
      try {
        idx0=Integer.parseInt(v.getLabel());
      } catch (Exception ex) {}
      if (idx0<0)
        continue;
      if (unionIds==null) {
        boolean found=false;
        for (int j = 0; j < explanations.size() && !found; j++)
          if (idx0==explanations.get(j).numId) {
            idx0=j;
            found=true;
          }
        if (!found)
          continue;
      }
      Iterator<Map.Entry<Vertex,Edge>> it = v.getEdges().entrySet().iterator();
      while (it.hasNext()) {
        Map.Entry<Vertex, Edge> pair = it.next();
        if (pair.getValue().isIncluded()) {
          Vertex v2=pair.getKey();
          int idx2=-1;
          try {
            idx2=Integer.parseInt(v2.getLabel());
          } catch (Exception ex) {}
          if (idx2>=0) {
            if (unionIds==null) {
              boolean found=false;
              for (int j = 0; j < explanations.size() && !found; j++)
                if (idx2==explanations.get(j).numId) {
                  idx2=j;
                  found=true;
                }
              if (!found)
                continue;
            }
            drawLinkBetweenPoints(gr, idx0, idx2);
          }
        }
      }
    }
  }
  
  public void drawLinkBetweenPoints(Graphics gr, int idx1, int idx2) {
    gr.drawLine(px[idx1],py[idx1],px[idx2],py[idx2]);
  }
 
  public String getToolTipText(MouseEvent me) {
    if (!isShowing())
      return null;
    if (me.getButton() != MouseEvent.NOBUTTON)
      return null;
    int idx=getPointIndexAtPosition(me.getX(),me.getY(),dotRadius);
    if (idx<0) {
      if (features==null || features.isEmpty())
        return null;
      String txt="<html><body style=background-color:rgb(255,255,204)>";
      txt+="<p align=center><b>Features used in computing distances:</b></p> ";
      for (int i=0; i<features.size(); i++)
        txt+=features.get(i)+"<br> ";
      txt+="</body></html>";
      return txt;
    }
    CommonExplanation ce=explanations.get(idx);
    Vector<CommonExplanation> vce=null;
    if (selected!=null && selected.size()>0) {
      vce=new Vector<>(selected.size());
      for (int i = 0; i < selected.size(); i++)
        vce.add(explanations.get(selected.get(i)));
    }
    try {
      BufferedImage bi = ShowSingleRule.getImageForRule(300,100, ce, vce, attrs, minmax);
      File outputfile = new File("img.png");
      ImageIO.write(bi, "png", outputfile);
      //System.out.println("img.png");
    } catch (IOException ex) { System.out.println("* error while writing image to file: "+ex.toString()); }
    return explanations.get(idx).toHTML(null,attrMinMax,"","img.png");
  }
  
  public static Color getColorForAction(int action, int minAction, int maxAction) {
    if (minAction==maxAction || action < minAction || action>maxAction)
      return Color.gray;
    float actionRatio = ((float)maxAction-action) / (maxAction-minAction);
    Color color = Color.getHSBColor(actionRatio * (hsbBlue[0] - hsbRed[0]),1,1);
    return new Color(color.getRed(),color.getGreen(),color.getBlue(),100);
  }
  
  public static Color getColorForQ(double q, double minQ, double maxQ) {
    if (Double.isNaN(minQ) || Double.isNaN(maxQ) || Double.isNaN(q) || minQ>=maxQ || q<minQ || q>maxQ)
      return Color.darkGray;
    float ratio=(float)((maxQ-q)/(maxQ-minQ));
    Color color = Color.getHSBColor(ratio * (hsbBlue[0] - hsbRed[0]),1,1);
    return new Color(color.getRed(),color.getGreen(),color.getBlue(),100);
  }
  
  public void selectLinkedToSelected() {
    if (unionIds==null || selector==null)
      return;
    ArrayList selected=selector.getSelected();
    if (selected==null || selected.isEmpty())
      return;
    ArrayList toAdd=new ArrayList(selected.size()*10);
    for (Object s:selected) {
      int idx=(Integer)s, unionIdx=unionIds[idx];
      for (int j=0; j<unionIds.length; j++)
        if (j!=idx && unionIdx==unionIds[j] && !selected.contains(j) && !toAdd.contains(j))
          toAdd.add(j);
    }
    if (!toAdd.isEmpty())
      selector.select(toAdd);
  }
  
  public ArrayList<Integer> getUnionIdsOfSelected() {
    if (unionIds==null || selector==null)
      return null;
    ArrayList selected=selector.getSelected();
    if (selected==null || selected.isEmpty())
      return null;
    ArrayList uIds=new ArrayList(selected.size()*10);
    for (Object s:selected) {
      int idx=(Integer)s;
      if (!uIds.contains(unionIds[idx]))
        uIds.add(unionIds[idx]);
    }
    if (uIds.isEmpty())
      return null;
    return uIds;
  }
}
