package TapasExplTreeViewer.vis;

import TapasDataReader.CommonExplanation;
import TapasExplTreeViewer.rules.RuleMaster;
import TapasExplTreeViewer.rules.UnitedRule;
import TapasExplTreeViewer.ui.ShowSingleRule;
import TapasUtilities.ItemSelectionManager;
import TapasUtilities.SingleHighlightManager;

import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.Vector;

public class RuleHierarchyVis extends JPanel
    implements ChangeListener, MouseListener, MouseMotionListener {
  
  protected static int minFeatureW = 15, minGlyphH = 50, xSpace = 30, ySpace = 15,
      xMargin = 25, yMargin = 15;
  public static float hsbRed[] = Color.RGBtoHSB(255, 0, 0, null);
  public static float hsbBlue[] = Color.RGBtoHSB(0, 0, 255, null);
  public static Color hlColor = new Color(255, 255, 0, 64);
  
  /**
   * A derived rule whose derivation hierarchy needs to be visualised.
   */
  public UnitedRule topRule = null;
  /**
   * The ranges of feature values
   */
  public Hashtable<String, float[]> attrMinMax = null;
  /**
   * The very original rule set (before any transformations have been applied)
   */
  public ArrayList<CommonExplanation> origRules =null;
  
  protected int minAction = Integer.MAX_VALUE, maxAction = Integer.MIN_VALUE;
  protected double minQValue = Double.NaN, maxQValue = Double.NaN;
  /**
   * used for rendering glyphs
   */
  protected Vector<String> attrs = null;
  protected Vector<float[]> minmax = null;
  
  /**
   * All the rules from the whole hierarchy of the top rule.
   */
  protected UnitedRule rules[] = null;
  /**
   * Indexes of rule parents in the hierarchy
   */
  protected int parentIdx[] = null;
  /**
   * For each rule in the array, the x- and y-coordinates (column and row)
   * in a 2-dimensional layout.
   */
  protected Point ruleXY[] = null;
  protected int maxRuleX = -1, maxRuleY = -1;
  /**
   * Glyph boundaries
   */
  protected Rectangle gBounds[] = null;
  /**
   * Used to speed up redrawing
   */
  protected BufferedImage off_Image = null;
  protected boolean off_Valid = false;
  /**
   * Highlighting and selection
   */
  protected SingleHighlightManager highlighter = null;
  protected ItemSelectionManager selector = null;
  public int hlIdx = -1;
  
  public RuleHierarchyVis(UnitedRule topRule, ArrayList origRules,
                          Vector<String> attrNamesOrdered,
                          Hashtable<String, float[]> attrMinMax) {
    this.topRule = topRule;
    this.origRules=origRules;
    this.attrMinMax = attrMinMax;
    attrs = attrNamesOrdered;
    
    if (attrs != null && attrMinMax != null) {
      minmax = new Vector<float[]>(attrs.size());
      for (int i = 0; i < attrs.size(); i++)
        minmax.add(attrMinMax.get(attrs.elementAt(i)));
    }
    for (int i = 0; i < origRules.size(); i++) {
      CommonExplanation ex = (CommonExplanation) origRules.get(i);
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
  
    ArrayList<UnitedRule> hList = topRule.putHierarchyInList(null);
    rules = hList.toArray(new UnitedRule[hList.size()]);
    ruleXY = new Point[rules.length];
    parentIdx = new int[rules.length];
    for (int i=0; i<rules.length; i++) {
      ruleXY[i]=null;
      parentIdx[i]=-1;
    }
    putSubTree(0, 0, 0, hList);
  
    highlighter = new SingleHighlightManager();
    highlighter.addChangeListener(this);
    selector = new ItemSelectionManager();
    selector.addChangeListener(this);
  
    addMouseMotionListener(this);
    addMouseListener(this);
  
    ToolTipManager.sharedInstance().registerComponent(this);
    ToolTipManager.sharedInstance().setDismissDelay(Integer.MAX_VALUE);
  }
  
  protected int putSubTree(int rIdx, int col, int row, ArrayList<UnitedRule> hList) {
    if (ruleXY[rIdx]!=null)
      return row;
    if (maxRuleX < col) maxRuleX = col;
    if (maxRuleY < row) maxRuleY = row;
    ruleXY[rIdx] = new Point(col, row);
    if (rules[rIdx].fromRules == null)
      return row;
    int lastRow = row;
    for (int i = 0; i < rules[rIdx].fromRules.size(); i++) {
      int chIdx = hList.indexOf(rules[rIdx].fromRules.get(i));
      if (chIdx < 0)
        continue;
      if (parentIdx[chIdx]>=0 || ruleXY[chIdx]!=null)
        continue;
      parentIdx[chIdx] = rIdx;
      lastRow = putSubTree(chIdx, col + 1, row, hList);
      row = lastRow + 1;
    }
    return lastRow;
  }
  
  public Dimension getPreferredSize() {
    if (attrs==null)
      return new Dimension(500,500);
    if (off_Image != null && off_Valid)
      return new Dimension(off_Image.getWidth(), off_Image.getHeight());
    int minGlyphW = attrs.size() * minFeatureW;
    return new Dimension(2 * xMargin + (1 + maxRuleX) * minGlyphW + maxRuleX * xSpace,
        2 * yMargin + (1 + maxRuleY) * minGlyphH + maxRuleY * ySpace);
  }
  
  public static Color getColorForAction(int action, int minAction, int maxAction) {
    if (minAction == maxAction || action < minAction || action > maxAction)
      return Color.gray;
    float actionRatio = ((float) maxAction - action) / (maxAction - minAction);
    Color color = Color.getHSBColor(actionRatio * (hsbBlue[0] - hsbRed[0]), 1, 1);
    return new Color(color.getRed(), color.getGreen(), color.getBlue(), 100);
  }
  
  public static Color getColorForQ(double q, double minQ, double maxQ) {
    if (Double.isNaN(minQ) || Double.isNaN(maxQ) || Double.isNaN(q) || minQ >= maxQ || q < minQ || q > maxQ)
      return Color.darkGray;
    float ratio = (float) ((maxQ - q) / (maxQ - minQ));
    Color color = Color.getHSBColor(ratio * (hsbBlue[0] - hsbRed[0]), 1, 1);
    return new Color(color.getRed(), color.getGreen(), color.getBlue(), 100);
  }
  
  public void drawRuleGlyph(Graphics2D gr, int ruleIdx,
                            Vector<CommonExplanation> exSelected,
                            int x, int y, int w, int h) {
    if (gr == null || ruleIdx < 0 || ruleIdx >= rules.length)
      return;
    UnitedRule r=rules[ruleIdx];
    BufferedImage ruleImage = ShowSingleRule.getImageForRule(w, h, r, exSelected, attrs, minmax);
    if (ruleImage == null)
      return;
    gr.drawImage(ruleImage, x, y, null);
    
    Color c = (minAction < maxAction) ? getColorForAction(r.action, minAction, maxAction) :
                  getColorForQ(r.meanQ, minQValue, maxQValue);
    gr.setColor(c);
    Stroke str = gr.getStroke();
    gr.setStroke(new BasicStroke(2));
    gr.drawRect(x, y, w, h);
  
    if (selector!=null && selector.isSelected(ruleIdx)) {
      gr.setColor(Color.black);
      gr.drawRect(x - 2, y - 2, w + 4, h + 4);
    }
    gr.setStroke(str);
    
    if (r.nOrigRight+r.nOrigWrong>0 || r.nCasesRight+r.nCasesWrong>0) {
      gr.setColor(new Color(0,0,0,160));
      if (r.nOrigRight+r.nOrigWrong>0) {
        float acc=1.0f*r.nOrigRight/(r.nOrigRight+r.nOrigWrong);
        int l=Math.round(acc*w);
        gr.drawRect(x,y-2,l,4);
        if (acc<1) {
          String txt = String.format("%.2f", acc);
          gr.drawString(txt, x + l - gr.getFontMetrics().stringWidth(txt), y - 2);
        }
      }
      if (r.nCasesRight+r.nCasesWrong>0) {
        float acc=1.0f*r.nCasesRight/(r.nCasesRight+r.nCasesWrong);
        int l=Math.round(acc*w);
        gr.drawRect(x,y+h-2,l,4);
        if (acc<1) {
          String txt = String.format("%.2f", acc);
          gr.drawString(txt, x + l - gr.getFontMetrics().stringWidth(txt), y + h + gr.getFontMetrics().getAscent());
        }
      }
    }
    
    if (gBounds == null) {
      gBounds = new Rectangle[rules.length];
      for (int i = 0; i < gBounds.length; i++)
        gBounds[i] = new Rectangle(0, 0, 0, 0);
    }
    Rectangle rect = gBounds[ruleIdx];
    rect.x = x - xSpace / 2 + 1;
    rect.y = y - ySpace / 2 + 1;
    rect.width = w + xSpace;
    rect.height = h + ySpace;
    
    if (r.numId >= 0) {
      String txt = Integer.toString(r.numId);
      gr.setColor(Color.darkGray);
      gr.drawString(txt, x - gr.getFontMetrics().stringWidth(txt) - 5, y + gr.getFontMetrics().getAscent());
    }
  }
  
  public void redraw() {
    if (isShowing())
      paintComponent(getGraphics());
  }
  
  public void drawHighlighted(Graphics gr) {
    if (gBounds == null || hlIdx < 0 || hlIdx > gBounds.length)
      return;
    gr.setColor(hlColor);
    Rectangle r = gBounds[hlIdx];
    gr.fillRect(r.x, r.y, r.width, r.height);
  }
  
  public void paintComponent(Graphics gr) {
    if (rules == null || attrs == null || attrs.isEmpty())
      return;
    if (gr == null)
      return;
    int w = getWidth(), h = getHeight();
    if (w < 1 || h < 1)
      return;
    
    //determine glyph dimensions
    int glyphW = (w - 2 * xMargin + xSpace) / (1 + maxRuleX) - xSpace;
    if (glyphW < 0.25 * attrs.size() * minFeatureW)
      return;
    int totalW = xMargin * 2 + (1 + maxRuleX) * (glyphW + xSpace) - xSpace;
    int glyphH = (h - 2 * yMargin + ySpace) / (1 + maxRuleY) - ySpace;
    if (glyphH < 10)
      return;
    ;
    int totalH = 2 * yMargin + (1 + maxRuleY) * (glyphH + ySpace) - ySpace;
    
    if (off_Image != null && off_Valid) {
      if (off_Image.getWidth() != getWidth() || off_Image.getHeight() != getHeight()) {
        off_Image = null;
        off_Valid = false;
      }
      else {
        gr.drawImage(off_Image, 0, 0, null);
        drawHighlighted(gr);
        return;
      }
    }
    
    if (off_Image == null)
      off_Image = new BufferedImage(getWidth(), getHeight(), BufferedImage.TYPE_INT_ARGB);
    Graphics2D g = off_Image.createGraphics();
    
    g.setColor(getBackground());
    g.fillRect(0, 0, getWidth() + 1, getHeight() + 1);
    
    ArrayList selected = (selector == null) ? null : selector.getSelected();
    Vector<CommonExplanation> exSelected = (selected == null || selected.isEmpty()) ? null :
                                               new Vector<CommonExplanation>(selected.size());
    if (exSelected != null) {
      for (int i = 0; i < selected.size(); i++) {
        int idx = (Integer) selected.get(i);
        if (idx >= 0 && idx < rules.length)
          exSelected.addElement(rules[idx]);
      }
    }
    
    for (int i = 0; i < rules.length; i++) {
      int x = xMargin + ruleXY[i].x * (glyphW + xSpace), y = yMargin + ruleXY[i].y * (glyphH + ySpace);
      drawRuleGlyph(g, i, exSelected, x, y, glyphW, glyphH);
      if (i > 0) {
        int j = parentIdx[i];
        y += glyphH / 2;
        int x0 = xMargin + ruleXY[j].x * (glyphW + xSpace) + glyphW,
            y0 = yMargin + ruleXY[j].y * (glyphH + ySpace) + glyphH / 2;
        g.setColor(Color.gray);
        g.drawLine(x0, y0, x0 + xSpace / 2, y0);
        g.drawLine(x0 + xSpace / 2, y0, x0 + xSpace / 2, y);
        g.drawLine(x0 + xSpace / 2, y, x, y);
      }
    }
    
    off_Valid = true;
    
    gr.drawImage(off_Image, 0, 0, null);
    drawHighlighted(gr);
  }
  
  
  public int getRuleIdxAtPosition(int x, int y) {
    if (gBounds == null)
      return -1;
    for (int i = 0; i < gBounds.length; i++)
      if (gBounds[i].contains(x, y))
        return i;
    return -1;
  }
  
  
  public String getToolTipText(MouseEvent me) {
    if (!isShowing())
      return null;
    if (me.getButton() != MouseEvent.NOBUTTON)
      return null;
    int idx = getRuleIdxAtPosition(me.getX(), me.getY());
    if (idx < 0)
      return null;
    ArrayList selIdxs=(selector==null)?null:selector.getSelected();
    Vector<CommonExplanation> exSelected=(selIdxs==null || selIdxs.isEmpty())?null:
                                             new Vector<CommonExplanation>(selIdxs.size());
    if (exSelected!=null) {
      for (int i = 0; i < selIdxs.size(); i++) {
        int ii=(Integer)selIdxs.get(i);
        if (ii>=0 && ii<rules.length) {
          exSelected.addElement(rules[ii]);
        }
      }
    }
    try {
      BufferedImage bi = ShowSingleRule.getImageForRule(300,100, rules[idx], exSelected, attrs, minmax);
      File outputfile = new File("img.png");
      ImageIO.write(bi, "png", outputfile);
      //System.out.println("img.png");
    } catch (IOException ex) { System.out.println("* error while writing image to file: "+ex.toString()); }
    return rules[idx].toHTML(null,attrMinMax,"","img.png");
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
        off_Valid=false;
        redraw();
      }
  }
  
  
  public void mouseExited(MouseEvent e) {
    highlighter.clearHighlighting();
  }
  
  public void mouseEntered(MouseEvent e) { }
  
  public void mouseMoved(MouseEvent e) {
    if (e.getButton() != MouseEvent.NOBUTTON)
      return;
    int idx=getRuleIdxAtPosition(e.getX(),e.getY());
    if (idx<0)
      highlighter.clearHighlighting();
    else
      highlighter.highlight(new Integer(idx));
  }
  public void mouseDragged(MouseEvent e) {}
  
  public void mousePressed(MouseEvent e) {
    if (e.getButton()>MouseEvent.BUTTON1) {
      int idx=getRuleIdxAtPosition(e.getX(),e.getY());
      if (idx>=0 && rules[idx].nOrigWrong>0) {
        JPopupMenu popup=new JPopupMenu();
        JMenuItem mit=new JMenuItem("Show wrong coverages of this rule");
        mit.addActionListener(new ActionListener() {
          @Override
          public void actionPerformed(ActionEvent e) {
            ArrayList<CommonExplanation> wrong=rules[idx].extractWrongCoverages(origRules,
                minAction >= maxAction);
            if (wrong==null || wrong.isEmpty()) {
              JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
                  "No wrong coverages found!",
                  "No exceptions", JOptionPane.INFORMATION_MESSAGE);
              return;
            }
            wrong.add(0,rules[idx]);
            RuleSetVis vis=new RuleSetVis(wrong,origRules,false,attrs,attrMinMax);
            vis.setOrigExList(origRules);
            Dimension size=Toolkit.getDefaultToolkit().getScreenSize();
            JFrame plotFrame=new JFrame("Wrong coverages of the rule that is shown first");
            plotFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            plotFrame.getContentPane().add(vis);
            plotFrame.pack();
            plotFrame.setSize((int)Math.min(plotFrame.getWidth(),0.7*size.width),
                (int)Math.min(plotFrame.getHeight(),0.7*size.height));
            plotFrame.setLocation(gBounds[idx].x+gBounds[idx].width,gBounds[idx].y+gBounds[idx].height);
            plotFrame.setVisible(true);
          }
        });
        popup.add(mit);
        popup.add(new JMenuItem("Cancel"));
        popup.show(this,e.getX(),e.getY());
      }
    }
  }
  
  public void mouseReleased(MouseEvent e) {}
  
  public void mouseClicked(MouseEvent e) {
    if (e.getClickCount()>1)
      selector.deselectAll();
    else
      if (e.getButton()==MouseEvent.BUTTON1){
        int idx=getRuleIdxAtPosition(e.getX(),e.getY());
        if (idx>=0)
          if (selector.isSelected(idx))
            selector.deselect(idx);
          else
            selector.select(idx);
      }
  }
}
