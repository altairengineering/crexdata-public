package TapasExplTreeViewer.vis;

import TapasDataReader.CommonExplanation;
import TapasExplTreeViewer.clustering.ObjectWithMeasure;
import TapasExplTreeViewer.rules.UnitedRule;
import TapasExplTreeViewer.ui.RulesOrderer;
import TapasExplTreeViewer.ui.ShowSingleRule;
import TapasUtilities.ItemSelectionManager;
import TapasUtilities.SingleHighlightManager;

import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Hashtable;
import java.util.Vector;

/**
 * A panel with multiple rules represented by glyphs.
 */
public class RuleSetVis extends JPanel
    implements ChangeListener, MouseListener, MouseMotionListener {
  
  protected static int minFeatureW=15, minGlyphH=50, xSpace=30, ySpace=10,
      xMargin=25, yMargin=10;
  public static float hsbRed[]=Color.RGBtoHSB(255,0,0,null);
  public static float hsbBlue[]=Color.RGBtoHSB(0,0,255,null);
  public static Color hlColor=new Color(255,255, 0,64);
  /**
   * The rules or explanations to be visualized. The elements are instances of
   * CommonExplanation or UnitedRule.
   */
  public ArrayList exList=null, fullExList=null, origExList=null;
  /**
   * If exList contains a subset of fullExList, ths array contains the
   * indices of the elements of fullExList in the smaller list or -1 when absent.
   */
  protected int idxInSubList[]=null;
  /**
   * The ranges of feature values
   */
  public Hashtable<String,float[]> attrMinMax=null;
  
  protected int minAction=Integer.MAX_VALUE, maxAction=Integer.MIN_VALUE;
  protected double minQValue=Double.NaN, maxQValue=Double.NaN;
  /**
   * used for rendering glyphs
   */
  protected Vector<String> attrs=null;
  protected ArrayList<String> listOfFeatures=null;
  protected Vector<float[]> minmax=null;
  /**
   * Ordering of glyphs
   */
  protected RulesOrderer rulesOrderer=null;
  protected int order[]=null;
  boolean orderedByDistances=false;
  /**
   * Highlighting and selection
   */
  protected SingleHighlightManager highlighter=null;
  protected ItemSelectionManager selector=null;
  public int hlIdx=-1;
  public ArrayList<Integer> selected=null;
  public boolean applyHiAndSelToFullList=true;
  /**
   * Glyph boundaries
   */
  protected Rectangle gBounds[]=null;
  /**
   * Used to speed up redrawing
   */
  protected BufferedImage off_Image=null;
  protected boolean off_Valid=false;
  
  
  public RuleSetVis(ArrayList exList, ArrayList fullExList,
                    Vector<String> attrNamesOrdered, Hashtable<String,float[]> attrMinMax) {
    this(exList,fullExList,true,attrNamesOrdered,attrMinMax);
  }
  
  public RuleSetVis(ArrayList exList, ArrayList fullExList, boolean applyHiAndSelToFullList,
                    Vector<String> attrNamesOrdered, Hashtable<String,float[]> attrMinMax) {
    this.exList=exList; this.fullExList=fullExList;
    this.attrMinMax=attrMinMax;
    attrs=attrNamesOrdered;
    if (attrs!=null && attrMinMax!=null) {
      minmax=new Vector<float[]>(attrs.size());
      listOfFeatures=new ArrayList<String>(attrs.size());
      for (int i=0; i<attrs.size(); i++) {
        minmax.add(attrMinMax.get(attrs.elementAt(i)));
        listOfFeatures.add(attrs.elementAt(i));
      }
    }
    this.applyHiAndSelToFullList=applyHiAndSelToFullList;
    if (applyHiAndSelToFullList && fullExList!=null && fullExList.size()>exList.size()) {
      idxInSubList=new int[fullExList.size()];
      for (int i=0; i<fullExList.size(); i++)
        idxInSubList[i]=exList.indexOf(fullExList.get(i));
    }
    addMouseListener(this);
    addMouseMotionListener(this);
    ToolTipManager.sharedInstance().registerComponent(this);
    ToolTipManager.sharedInstance().setDismissDelay(Integer.MAX_VALUE);
    highlighter=new SingleHighlightManager();
    highlighter.addChangeListener(this);
    selector=new ItemSelectionManager();
    selector.addChangeListener(this);
  }
  
  public void setOrigExList(ArrayList origExList) {
    this.origExList = origExList;
  }
  
  protected void determineMinMaxQorAction() {
    ArrayList rules=(origExList!=null)?origExList:(fullExList!=null)?fullExList:exList;
    for (int i=0; i<rules.size(); i++) {
      CommonExplanation ex = (CommonExplanation) rules.get(i);
      if (minAction > ex.action)
        minAction = ex.action;
      if (maxAction < ex.action)
        maxAction = ex.action;
      if (!Double.isNaN(ex.meanQ)) {
        if (Double.isNaN(minQValue) || minQValue > ex.minQ)
          minQValue = ex.minQ;
        if (Double.isNaN(maxQValue) || maxQValue < ex.maxQ)
          maxQValue = ex.maxQ;
      }
    }
  }
  
  public SingleHighlightManager getHighlighter(){
    return highlighter;
  }
  
  public ItemSelectionManager getSelector() {
    return selector;
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
    if (selector!=null) {
      selector.addChangeListener(this);
      getLocalSelection();
    }
  }
  
  public void setRulesOrderer(RulesOrderer rulesOrderer) {
    this.rulesOrderer = rulesOrderer;
    if (rulesOrderer!=null)
      order=rulesOrderer.getRulesOrder(exList);
  }
  
  public void redraw(){
    if (isShowing())
      paintComponent(getGraphics());
  }
  
  public void drawHighlighted(Graphics gr) {
    if (gBounds==null || hlIdx<0 || hlIdx>gBounds.length)
      return;
    gr.setColor(hlColor);
    Rectangle r=gBounds[hlIdx];
    gr.fillRect(r.x,r.y,r.width,r.height);
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
  
  public void drawRuleGlyph(Graphics2D gr, int ruleIdx,
                            Vector<CommonExplanation> exSelected,
                            int x, int y, int w, int h) {
    if (gr==null || ruleIdx<0 || ruleIdx>=exList.size())
      return;
    CommonExplanation ex=(CommonExplanation)exList.get(ruleIdx);
    BufferedImage ruleImage= ShowSingleRule.getImageForRule(w,h,ex,exSelected,attrs,minmax);
    if (ruleImage==null)
      return;
    gr.drawImage(ruleImage,x,y,null);
    Color c=(minAction<maxAction)?getColorForAction(ex.action,minAction,maxAction):
                getColorForQ(ex.meanQ,minQValue,maxQValue);
    gr.setColor(c);
    Stroke str=gr.getStroke();
    gr.setStroke(new BasicStroke(2));
    gr.drawRect(x,y,w,h);
    if (selected!=null && selected.contains(ruleIdx)) {
      gr.setColor(Color.black);
      gr.drawRect(x-2,y-2,w+4,h+4);
    }
    gr.setStroke(str);

    if (gBounds==null) {
      gBounds=new Rectangle[exList.size()];
      for (int i=0; i<gBounds.length; i++)
        gBounds[i]=new Rectangle(0,0,0,0);
    }
    Rectangle r=gBounds[ruleIdx];
    r.x=x-xSpace/2+1; r.y=y-ySpace/2+1; r.width=w+xSpace; r.height=h+ySpace;

    if (ex.numId>=0) {
      String txt=Integer.toString(ex.numId);
      gr.setColor(Color.darkGray);
      gr.drawString(txt, x - gr.getFontMetrics().stringWidth(txt)-5, y + gr.getFontMetrics().getAscent());
    }
  }
  
  public Dimension getPreferredSize() {
    if (off_Image!=null && off_Valid)
      return new Dimension(off_Image.getWidth(),off_Image.getHeight());
    if (exList==null || exList.isEmpty())
      return new Dimension(100,50);
    int minGlyphW=attrs.size()*minFeatureW;
    int nGlyphsInRow=Math.min(exList.size(),5),
        nRows=(int)Math.round(Math.ceil(1.0*exList.size()/nGlyphsInRow));
    return new Dimension(2 * xMargin + nGlyphsInRow * minGlyphW + (nGlyphsInRow-1) * xSpace,
        2*yMargin+nRows*minGlyphH+(nRows-1)*ySpace);
  }
  
  public void paintComponent(Graphics gr) {
    if (exList==null || exList.isEmpty() || attrs==null || attrs.isEmpty())
      return;
    if (gr==null)
      return;
    int w=getWidth(), h=getHeight();
    if (w<1 || h<1)
      return;
    
    if (minAction>maxAction && Double.isNaN(maxQValue))
      determineMinMaxQorAction();
    
    if (getParent() instanceof JViewport) {
      JViewport vp=(JViewport)getParent();
      if (w>vp.getWidth())
        w=vp.getWidth();
      if (h>vp.getHeight())
        h=vp.getHeight();
    }
  
    //determine glyph dimensions
    int glyphW=attrs.size()*minFeatureW;
    int nGlyphsInRow=Math.max(1,Math.min(exList.size(),(w-2*xMargin+xSpace)/(glyphW+xSpace)));
    glyphW=(w-2*xMargin)/nGlyphsInRow-xSpace;
    int totalW=xMargin*2+nGlyphsInRow*(glyphW+xSpace)-xSpace;
    int nRows=(int)Math.round(Math.ceil(1.0*exList.size()/nGlyphsInRow));
    if (nRows*nGlyphsInRow<exList.size())
      ++nRows;
    int glyphH=Math.max((h-2*yMargin+ySpace)/nRows-ySpace,minGlyphH);
    int totalH=2*yMargin+nRows*(glyphH+ySpace)-ySpace;

    if (totalH>h && totalH!=getHeight()) {
      off_Valid=false; off_Image=null;
      setSize(Math.max(w, totalW), totalH);
      paintComponent(gr);
      return;
    }

    if (off_Image!=null && off_Valid) {
      if (off_Image.getWidth()!=getWidth() || off_Image.getHeight()!=getHeight()) {
        off_Image = null; off_Valid=false;
      }
      else {
        gr.drawImage(off_Image,0,0,null);
        drawHighlighted(gr);
        return;
      }
    }
  
    if (off_Image==null)
      off_Image=new BufferedImage(getWidth(),getHeight(),BufferedImage.TYPE_INT_ARGB);
    Graphics2D g = off_Image.createGraphics();
  
    g.setColor(getBackground());
    g.fillRect(0,0,getWidth()+1,getHeight()+1);
  
    ArrayList fullSelected=(selector==null)?null:selector.getSelected();
    Vector<CommonExplanation> exSelected=(fullSelected==null || fullSelected.isEmpty())?null:
                                             new Vector<CommonExplanation>(fullSelected.size());
    if (exSelected!=null) {
      ArrayList rules=(applyHiAndSelToFullList && fullExList!=null)?fullExList:exList;
      for (int i = 0; i < fullSelected.size(); i++) {
        int idx=(Integer)fullSelected.get(i);
        if (idx>=0 && idx<rules.size()) {
          CommonExplanation ex=(CommonExplanation) rules.get(idx);
          exSelected.addElement(ex);
        }
      }
    }
    
    int x=xMargin, y=yMargin, nInRow=0;
    for (int i=0; i<exList.size(); i++) {
      drawRuleGlyph(g,(order==null)?i:order[i],exSelected,x,y,glyphW,glyphH);
      ++nInRow;
      if (i+1<exList.size())
        if (nInRow<nGlyphsInRow)
          x+=glyphW+xSpace;
        else {
          x=xMargin; y+=glyphH+ySpace; nInRow=0;
        }
    }
  
    off_Valid=true;
    
    gr.drawImage(off_Image,0,0,null);
    drawHighlighted(gr);
  }
  
  public void stateChanged(ChangeEvent e) {
    if (e.getSource().equals(highlighter)) {
      if (!off_Valid)
        return;
      int idx=(highlighter.highlighted!=null &&
                   (highlighter.highlighted instanceof Integer))?(Integer)highlighter.highlighted:-1;
      if (idx>=0 && idxInSubList!=null)
        idx=idxInSubList[idx];
      if (hlIdx!=idx) {
        hlIdx=idx;
        if (off_Valid)
          redraw();
      }
    }
    else
      if (e.getSource().equals(selector)) {
        getLocalSelection();
        off_Valid=false;
        redraw();
      }
  }
  
  protected void getLocalSelection() {
    ArrayList currSel=selector.getSelected();
    if (currSel!=null && !currSel.isEmpty() && idxInSubList!=null) {
      ArrayList<Integer> localSel=new ArrayList<Integer>(currSel.size());
      for (int i=0; i<currSel.size(); i++)
        if (currSel.get(i) instanceof Integer) {
          int idx=(Integer) currSel.get(i);
          if (idx>=0 && idx<idxInSubList.length) {
            idx=idxInSubList[idx];
            if (idx>=0)
              localSel.add(idx);
          }
        }
      currSel=localSel;
    }
    if (currSel==null || currSel.isEmpty()) {
      if (selected!=null)
        selected.clear();
    }
    else {
      if (selected==null)
        selected=new ArrayList<Integer>(100);
      selected.clear();
      for (int i=0; i<currSel.size(); i++)
        if (currSel.get(i) instanceof Integer)
          selected.add((Integer)currSel.get(i));
    }
  }
  
  public int getRuleIdxAtPosition(int x, int y) {
    if (gBounds==null)
      return -1;
    for (int i=0; i<gBounds.length; i++)
      if (gBounds[i].contains(x,y))
        return i;
    return -1;
  }
  
  public void mousePressed(MouseEvent e){}
  
  public void mouseReleased(MouseEvent e){}
  
  public void mouseClicked(MouseEvent e) {
    if (e.getClickCount()==2) {
      if (e.getButton()==MouseEvent.BUTTON1) {
        int idx = getRuleIdxAtPosition(e.getX(), e.getY());
        if (idx < 0)
          selector.deselectAll();
      }
      else {
        if (order==null)
          return;
        order=(rulesOrderer != null)?rulesOrderer.getRulesOrder(exList):null;
        orderedByDistances=false;
        off_Valid=false;
        redraw();
      }
    }
    else
    if (e.getClickCount()==1) {
      int idx = getRuleIdxAtPosition(e.getX(), e.getY());
      if (idx < 0)
        return;
      if (e.getButton() == MouseEvent.BUTTON1) {
        if (idxInSubList != null)
          for (int i = 0; i < idxInSubList.length; i++)
            if (idxInSubList[i] == idx) {
              idx = i;
              break;
            }
        Integer iSel = new Integer(idx);
        if (selector.isSelected(iSel))
          selector.deselect(iSel);
        else
          selector.select(iSel);
      }
      else {
        if (orderedByDistances && order!=null && order[0]==idx)
          return;
        CommonExplanation ex=(CommonExplanation)exList.get(idx);
        ObjectWithMeasure rDist[]=new ObjectWithMeasure[exList.size()-1];
        int k=0;
        for (int i=0; i<exList.size(); i++)
          if (i!=idx)
            rDist[k++]=new ObjectWithMeasure(i, UnitedRule.distance(ex,(CommonExplanation)exList.get(i),attrMinMax));
        Arrays.sort(rDist);
        if (order==null)
          order=new int[exList.size()];
        order[0]=idx;
        for (int i=0; i<rDist.length; i++)
          order[i+1]=(Integer)rDist[i].obj;
        orderedByDistances=true;
        off_Valid=false;
        redraw();
      }
    }
  }
  
  public void mouseExited(MouseEvent e) {
    highlighter.clearHighlighting();
  }
  
  public void mouseEntered(MouseEvent e) { }
  
  public void mouseMoved(MouseEvent e) {
    int idx=getRuleIdxAtPosition(e.getX(),e.getY());
    if (idx<0)
      highlighter.clearHighlighting();
    else {
      if (idxInSubList!=null) {
        boolean found=false;
        for (int i = 0; i < idxInSubList.length && !found; i++)
          if (idxInSubList[i] == idx) {
            idx = i;
            found=true;
          }
        if (!found)
          return;
      }
      highlighter.highlight(new Integer(idx));
    }
  }
  public void mouseDragged(MouseEvent e) {}
  
  public String getToolTipText(MouseEvent me) {
    if (!isShowing())
      return null;
    if (me.getButton() != MouseEvent.NOBUTTON)
      return null;
    int idx=getRuleIdxAtPosition(me.getX(),me.getY());
    if (idx<0) {
      String txt="<html><body style=background-color:rgb(255,255,204)>";
      if (orderedByDistances && order!=null)
        txt+="Rule glyphs are ordered by distances to rule "+
                 ((CommonExplanation)exList.get(order[0])).numId+"<br><br>";
      txt+="Right mouse button click: order glyphs by distances to the rule at the mouse cursor<br>";
      txt+="Right mouse button double-click: order glyphs according to the current order of the table rows<br><br>";
      txt+="Left mouse button click: select/deselect<br>";
      txt+="Left mouse button double-click in empty space: deselect all<br>";
      txt+="</body></html>";
      return txt;
    }
    CommonExplanation ce=(CommonExplanation) exList.get(idx);
    ArrayList fullSelected=(selector==null)?null:selector.getSelected();
    Vector<CommonExplanation> exSelected=(fullSelected==null || fullSelected.isEmpty())?null:
                                             new Vector<CommonExplanation>(fullSelected.size());
    if (exSelected!=null) {
      ArrayList rules=(applyHiAndSelToFullList && fullExList!=null)?fullExList:exList;
      for (int i = 0; i < fullSelected.size(); i++) {
        int ii=(Integer)fullSelected.get(i);
        if (ii>=0 && ii<rules.size()) {
          CommonExplanation ex=(CommonExplanation) rules.get(ii);
          exSelected.addElement(ex);
        }
      }
    }
    try {
      BufferedImage bi = ShowSingleRule.getImageForRule(300,100, ce, exSelected, attrs, minmax);
      File outputfile = new File("img.png");
      ImageIO.write(bi, "png", outputfile);
      //System.out.println("img.png");
    } catch (IOException ex) { System.out.println("* error while writing image to file: "+ex.toString()); }
    return ce.toHTML(listOfFeatures,attrMinMax,"","img.png");
  }
}
