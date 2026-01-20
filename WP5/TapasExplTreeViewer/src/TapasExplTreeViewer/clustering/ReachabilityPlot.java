package TapasExplTreeViewer.clustering;

import TapasUtilities.ItemSelectionManager;
import TapasUtilities.SingleHighlightManager;
import it.unipi.di.sax.optics.ClusterObject;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.image.BufferedImage;
import java.util.ArrayList;

public class ReachabilityPlot extends JPanel
    implements ChangeListener, MouseListener, MouseMotionListener {
  public static int minBarW=2;
  public static Color color1=new Color(255,140,0),
    color2=new Color(255-color1.getRed(),255-color1.getGreen(), 255-color1.getBlue());
  public static Color highlightColor=new Color(255,0,0,160),
      highlightFillColor=new Color(255,255,0,100),
      selectColor=new Color(0,0,0,200);
  /**
   * Objects ordered by the clustering algorithm
   */
  protected ArrayList<ClusterObject> objOrdered = null;
  protected int origObjIndexesInOrder[]=null;
  
  public double maxDistance=Double.NaN;
  public double threshold=Double.NaN;
  public ClustersAssignments clAss=null;
  
  protected int barW=minBarW, xMarg=5, yMarg=5, plotW=0, plotH=0;
  protected double scale=Double.NaN;
  /**
   * Highlighting and selection
   */
  protected SingleHighlightManager highlighter=null;
  protected ItemSelectionManager selector=null;
  public int hlIdx=-1;
  public ArrayList<Integer> selected=null;
  /**
   * Used to speed up redrawing
   */
  protected BufferedImage off_Image=null, off_selected=null;
  protected boolean off_Valid=false, off_selected_Valid =false;
  
  public ReachabilityPlot (ArrayList<ClusterObject> objOrdered) {
    setObjectsOrder(objOrdered);
    addMouseListener(this);
    addMouseMotionListener(this);
    ToolTipManager.sharedInstance().registerComponent(this);
    ToolTipManager.sharedInstance().setDismissDelay(Integer.MAX_VALUE);
  }
  
  public void setObjectsOrder(ArrayList<ClusterObject> objOrdered) {
    this.objOrdered=objOrdered;
    if (objOrdered!=null && !objOrdered.isEmpty()) {
      if (!isShowing())
        setPreferredSize(new Dimension(Math.max(1200, minBarW * objOrdered.size()+10), 300));
      
      maxDistance=Double.NaN;
      threshold=Double.NaN;
      scale=Double.NaN;
      clAss=null;
      off_Valid=off_selected_Valid=false;
      
      origObjIndexesInOrder=new int[objOrdered.size()];
      for (int i=0; i<origObjIndexesInOrder.length; i++)
        origObjIndexesInOrder[i]=-1;
      for (int i=0; i<objOrdered.size(); i++) {
        ClusterObject clObj=objOrdered.get(i);
        int idx=getOrigObjIndex(clObj);
        if (idx>=0 && idx<origObjIndexesInOrder.length)
          origObjIndexesInOrder[idx]=i;
        double rd=clObj.getReachabilityDistance();
        if (!Double.isNaN(rd) && !Double.isInfinite(rd) && (Double.isNaN(maxDistance) || maxDistance<rd))
          maxDistance=rd;
      }
      if (isShowing())
        redraw();
    }
  }
  
  public int getOrigObjIndex(ClusterObject clObj) {
    if (clObj==null)
      return -1;
    if (clObj.getOriginalObject() instanceof Integer)
      return (Integer)clObj.getOriginalObject();
    if (clObj.getOriginalObject() instanceof ClusterObject)
      return getOrigObjIndex((ClusterObject)clObj.getOriginalObject());
    return -1;
  }
  
  public double getMaxDistance() {
    return maxDistance;
  }
  
  public void setThreshold(double threshold) {
    this.threshold = threshold;
    if (isShowing() && off_Image!=null && off_Valid &&
            !Double.isNaN(threshold) && threshold>0 && threshold<maxDistance) {
      Graphics g=getGraphics();
      g.drawImage(off_Image,0,0,null);
      int th=(int)Math.round(scale*threshold);
      g.setColor(Color.red);
      g.drawLine(0,plotH-yMarg-th,plotW,plotH-yMarg-th);
    }
  }
  
  public double getThreshold() {
    return threshold;
  }
  
  public void setCusterAssignments(ClustersAssignments clAss) {
    this.clAss=clAss;
    off_Valid=false;
    if (isShowing())
      redraw();
  }
  
  public void setHighlighter(SingleHighlightManager highlighter) {
    if (this.highlighter!=null)
      if (this.highlighter.equals(highlighter))
        return;
      else
        this.highlighter.removeChangeListener(this);
    this.highlighter = highlighter;
    if (highlighter!=null)
      highlighter.addChangeListener(this);
  }
  
  public void setSelector(ItemSelectionManager selector) {
    if (this.selector!=null)
      if (this.selector.equals(selector))
        return;
      else
        this.selector.removeChangeListener(this);
    this.selector = selector;
    if (selector!=null)
      selector.addChangeListener(this);
  }
  
  public int getOrigObjIdxAtPosition(int x, int y) {
    if (x<xMarg || x>plotW-xMarg || y<yMarg || y>plotH-yMarg)
      return -1;
    int idx=(x-xMarg)/barW;
    if (idx<0 || idx>=objOrdered.size())
      return -1;
    double rd = objOrdered.get(idx).getReachabilityDistance();
    int barH=(int)Math.round((Double.isNaN(rd) || Double.isInfinite(rd))?plotH-yMarg:scale*rd);
    if (y<plotH - yMarg - barH)
      return -1;
    return getOrigObjIndex(objOrdered.get(idx));
  }
  
  public void drawSelected(Graphics gr) {
    if (selected==null || selected.isEmpty() || objOrdered==null || Double.isNaN(maxDistance))
      return;
    if (plotW<1 || plotH<1)
      return;
    if (off_selected!=null && off_selected_Valid) {
      if (off_selected.getWidth()!=plotW || off_selected.getHeight()!=plotH) {
        off_selected = null; off_selected_Valid =false;
      }
      else {
        gr.drawImage(off_selected,0,0,null);
        return;
      }
    }
    
    off_selected=new BufferedImage(plotW,plotH,BufferedImage.TYPE_INT_ARGB);
    Graphics2D g = off_selected.createGraphics();

    for (int j=0; j<selected.size(); j++) {
      int i=selected.get(j);
      int idx=origObjIndexesInOrder[i];
      if (idx<0)
        continue;
      double rd = objOrdered.get(idx).getReachabilityDistance();
      int barH=(int)Math.round((Double.isNaN(rd) || Double.isInfinite(rd))?plotH-yMarg:scale*rd);
      if (barH>0) {
        g.setColor(selectColor);
        int x=xMarg+idx*barW;
        g.fillRect(x, plotH - yMarg - barH, barW, barH);
        g.drawRect(x, plotH - yMarg - barH, barW, barH);
      }
    }
    
    gr.drawImage(off_selected,0,0,null);
    off_selected_Valid=true;
  }
  
  public void drawHighlighted(Graphics gr) {
    if (hlIdx<0 || objOrdered==null || Double.isNaN(maxDistance))
      return;
    int idx=origObjIndexesInOrder[hlIdx];
    if (idx<0)
      return;
    double rd = objOrdered.get(idx).getReachabilityDistance();
    int barH=(int)Math.round((Double.isNaN(rd) || Double.isInfinite(rd))?plotH-yMarg:scale*rd);
    if (barH>0) {
      gr.setColor(highlightFillColor);
      int x=xMarg+idx*barW;
      gr.fillRect(x, plotH - yMarg - barH, barW, barH);
      gr.setColor(highlightColor);
      gr.drawRect(x, plotH - yMarg - barH, barW, barH);
    }
  }
  
  public static Color getColorForCluster(int cluster) {
    if (cluster<0)
      return Color.gray;
    return (cluster%2==0)?color1:color2;
  }
  
  public void paintComponent(Graphics gr) {
    if (gr==null)
      return;
    plotW=getWidth();
    plotH=getHeight();
    if (plotW<1 || plotH<1)
      return;
    scale=(plotH-10)/maxDistance;
    if (off_Image!=null && off_Valid) {
      if (off_Image.getWidth()!=plotW || off_Image.getHeight()!=plotH) {
        off_Image = null; off_Valid=false;
        off_selected_Valid=false;
      }
      else {
        gr.drawImage(off_Image,0,0,null);
        drawSelected(gr);
        drawHighlighted(gr);
        return;
      }
    }
  
    if (off_Image==null || off_Image.getWidth()!=plotW || off_Image.getHeight()!=plotH)
      off_Image=new BufferedImage(plotW,plotH,BufferedImage.TYPE_INT_ARGB);
    Graphics2D g = off_Image.createGraphics();
  
    g.setColor(getBackground());
    g.fillRect(0,0,plotW+1,plotH+1);
    
    if (objOrdered==null || Double.isNaN(maxDistance))
      return;
    
    barW=Math.max(minBarW,(plotW-10)/objOrdered.size());
    
    int x=xMarg;
    g.setColor(Color.gray);
    for (int i=0; i<objOrdered.size(); i++) {
      double rd = objOrdered.get(i).getReachabilityDistance();
      int barH=(int)Math.round((Double.isNaN(rd) || Double.isInfinite(rd))?plotH-yMarg:scale*rd);
      if (barH>0) {
        if (clAss!=null)
          g.setColor(getColorForCluster(clAss.clusters[i]));
        else
          g.setColor(Color.gray);
        g.fillRect(x, plotH - yMarg - barH, barW, barH);
      }
      x+=barW;
    }
  
    if (!Double.isNaN(threshold) && threshold>0 && threshold<maxDistance){
      int th=(int)Math.round(scale*threshold);
      g.setColor(Color.red);
      g.drawLine(0,plotH-yMarg-th,plotW,plotH-yMarg-th);
    }
  
    gr.drawImage(off_Image,0,0,null);
    off_Valid=true;
    drawSelected(gr);
    drawHighlighted(gr);
  }
  
  public void redraw(){
    if (isShowing())
      paintComponent(getGraphics());
  }
  
  public void stateChanged(ChangeEvent e) {
    if (e.getSource().equals(highlighter)) {
      if (!off_Valid)
        return;
      int idx=(highlighter.highlighted!=null &&
                   (highlighter.highlighted instanceof Integer))?(Integer)highlighter.highlighted:-1;
      if (hlIdx!=idx) {
        hlIdx=idx;
        if (off_Valid)
          redraw();
      }
    }
    else
      if (e.getSource().equals(selector)) {
        ArrayList currSel=selector.selected;
        if (ItemSelectionManager.sameContent(currSel,selected))
          return;
        if (currSel==null || currSel.isEmpty())
          selected.clear();
        else {
          if (selected==null)
            selected=new ArrayList<Integer>(100);
          selected.clear();
          for (int i=0; i<currSel.size(); i++)
            if (currSel.get(i) instanceof Integer)
              selected.add((Integer)currSel.get(i));
        }
        off_selected_Valid=false;
        if (off_Valid)
          redraw();
      }
  }
  
  protected int pressX=-1, dragX=-1;
  
  public void mousePressed(MouseEvent e) {
    if (e.getButton()==MouseEvent.BUTTON1)
      pressX=e.getX();
  }
  
  public void mouseReleased(MouseEvent e) {
    if (pressX>=0 && dragX>=0 && dragX!=pressX) {
      int x1=Math.min(pressX,dragX), x2=Math.max(pressX,dragX);
      int idx1=Math.max((x1-xMarg)/barW,0), idx2=Math.min((x2-xMarg)/barW,objOrdered.size()-1);
      if (idx1<=idx2) {
        ArrayList<Integer> indexes=new ArrayList<Integer>(idx2-idx1+1);
        for (int i=idx1; i<=idx2; i++)
          indexes.add(new Integer(getOrigObjIndex(objOrdered.get(i))));
        if (!indexes.isEmpty())
          if (selector.areAllSelected(indexes))
            selector.deselect(indexes);
          else
            selector.select(indexes);
      }
    }
    pressX=dragX=-1;
  }
  
  public void mouseClicked(MouseEvent e) {
    if (e.getClickCount()>1)
      selector.deselectAll();
    else
      if (e.getButton()==MouseEvent.BUTTON1){
        int origIdx=getOrigObjIdxAtPosition(e.getX(), e.getY());
        if (origIdx>=0) {
          int idx=new Integer(origIdx);
          if (selector.isSelected(idx))
            selector.deselect(idx);
          else
            selector.select(idx);
        }
      }
  }
  
  public void mouseExited(MouseEvent e) {
    highlighter.clearHighlighting();
  }
  
  public void mouseEntered(MouseEvent e) { }
  
  public void mouseMoved(MouseEvent e) {
    int idx=getOrigObjIdxAtPosition(e.getX(), e.getY());
    if (idx<0)
      highlighter.clearHighlighting();
    else
      highlighter.highlight(new Integer(idx));
  }
  public void mouseDragged(MouseEvent e) {
    if (pressX>=0){
      if (dragX>=0 && dragX!=pressX)
        redraw();
      dragX=e.getX();
      if (dragX!=pressX) {
        int x=Math.min(pressX,dragX), w=Math.abs(dragX-pressX);
        Graphics g = getGraphics();
        g.setColor(new Color(0,0,0,96));
        g.fillRect(x,0,w,plotH);
      }
    }
  }
  
  public String getToolTipText(MouseEvent me) {
    if (!isShowing())
      return null;
    if (me.getButton() != MouseEvent.NOBUTTON)
      return null;
    int origIdx=getOrigObjIdxAtPosition(me.getX(), me.getY());
    if (origIdx<0)
      return null;
    int idx=(me.getX()-xMarg)/barW;
  
    String txt="<html><body style=background-color:rgb(255,255,204)>";
    txt += "<table border=0 cellmargin=3 cellpadding=3 cellspacing=3>";
    txt+="<tr><td>Ordinal N</td><td>"+idx+"</td></tr>";
    txt+="<tr><td>Original object index</td><td>"+origIdx+"</td></tr>";
    txt+="<tr><td>Reachability distance</td><td>"+
             String.format("%.5f",objOrdered.get(idx).getReachabilityDistance())+"</td></tr>";
    txt += "</table>";
    txt+="</body></html>";
    return txt;
  }
}

