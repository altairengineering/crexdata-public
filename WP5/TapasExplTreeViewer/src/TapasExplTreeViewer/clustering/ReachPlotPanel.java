package TapasExplTreeViewer.clustering;

import TapasUtilities.ItemSelectionManager;
import TapasUtilities.SingleHighlightManager;
import it.unipi.di.sax.optics.ClusterObject;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;

public class ReachPlotPanel extends JPanel implements ChangeListener, ActionListener {
  /**
   * Runs the clustering algorithm (OPTICS)
   */
  public OPTICS_Runner optics=null;
  /**
   * Objects ordered by the clustering algorithm
   */
  protected ArrayList<ClusterObject> objOrdered = null;
  public ReachabilityPlot rPlot=null;
  
  protected JLabel labTop=null, labMaxTh=null;
  protected JSlider thresholdSlider=null;
  protected JTextField tfThreshold=null, tfRadius=null, tfNeighbors=null;
  
  public ReachPlotPanel(ArrayList<ClusterObject> objOrdered, OPTICS_Runner optics) {
    this.objOrdered=objOrdered;
    this.optics=optics;
    
    rPlot=new ReachabilityPlot(objOrdered);
    
    setLayout(new BorderLayout());
    JScrollPane scp=new JScrollPane(rPlot);
    scp.setVerticalScrollBarPolicy(ScrollPaneConstants.VERTICAL_SCROLLBAR_NEVER);
    add(scp,BorderLayout.CENTER);
    
    JPanel p=new JPanel(new GridLayout(0,1));
    add(p,BorderLayout.NORTH);
    if (optics!=null) {
      tfRadius=new JTextField(String.format("%.5f",optics.getNeibRadius(),10));
      tfNeighbors=new JTextField(String.valueOf(optics.getMinNeighbors()),3);
      JButton b=new JButton("Run again");
      b.setActionCommand("run");
      b.addActionListener(this);
      Panel pp = new Panel(new FlowLayout(FlowLayout.CENTER, 10, 0));
      pp.add(new JLabel("Neighbourhood radius:",JLabel.RIGHT));
      pp.add(tfRadius);
      pp.add(new JLabel("Minimal N of neighbours:",JLabel.RIGHT));
      pp.add(tfNeighbors);
      pp.add(new JLabel(" "));
      pp.add(b);
      p.add(pp);
    }
    labTop=new JLabel("Maximal reachability distance: "+
                          String.format("%.5f",rPlot.getMaxDistance()),JLabel.CENTER);
    p.add(labTop);
    
    tfThreshold=new JTextField(10);
    p=new JPanel(new FlowLayout(FlowLayout.CENTER,5,0));
    p.add(new JLabel("Distance threshold for clustering:"));
    p.add(tfThreshold);
    tfThreshold.addActionListener(this);
    
    thresholdSlider=new JSlider(JSlider.HORIZONTAL,1,1000,1000);
    JPanel bp=new JPanel(new BorderLayout());
    bp.add(p,BorderLayout.WEST);
    bp.add(thresholdSlider,BorderLayout.CENTER);
    labMaxTh=new JLabel("max = "+String.format("%.5f",rPlot.getMaxDistance()));
    bp.add(labMaxTh,BorderLayout.EAST);
    add(bp,BorderLayout.SOUTH);
    thresholdSlider.addChangeListener(this);
  }
  
  public void updateObjectsOrder(ArrayList<ClusterObject> objOrdered) {
    this.objOrdered=objOrdered;
    rPlot.setObjectsOrder(objOrdered);
    labTop.setText("Maximal reachability distance: "+
                          String.format("%.5f",rPlot.getMaxDistance()));
    labMaxTh.setText("max = "+String.format("%.5f",rPlot.getMaxDistance()));
    tfThreshold.setText("");
    thresholdSlider.removeChangeListener(this);
    thresholdSlider.setValue(thresholdSlider.getMaximum());
    thresholdSlider.addChangeListener(this);
    tfRadius.setText(String.format("%.5f",optics.getNeibRadius()));
    tfNeighbors.setText(Integer.toString(optics.getMinNeighbors()));
  }
  
  public void setHighlighter(SingleHighlightManager highlighter) {
    rPlot.setHighlighter(highlighter);
  }
  
  public void setSelector(ItemSelectionManager selector) {
    rPlot.setSelector(selector);
  }
  
  public void stateChanged(ChangeEvent e) {
    if (e.getSource().equals(thresholdSlider)) {
      double th=rPlot.getMaxDistance()*thresholdSlider.getValue()/thresholdSlider.getMaximum();
      rPlot.setThreshold(th);
      makeAndShowClusters();
      String str=String.format("%.5f",th);
      tfThreshold.setText(str);
    }
  }
  
  public void actionPerformed(ActionEvent e){
    if (e.getSource().equals(tfThreshold)) {
      double th=Double.NaN;
      try {
        th=Double.parseDouble(tfThreshold.getText());
      } catch (Exception ex) {}
      if (Double.isNaN(th) || th<=0 || th>=rPlot.getMaxDistance()) {
        String str=String.format("%.5f",rPlot.getThreshold());
        tfThreshold.setText(str);
      }
      else {
        rPlot.setThreshold(th);
        int sliderValue=(int)Math.round(th/rPlot.getMaxDistance()*thresholdSlider.getMaximum());
        thresholdSlider.removeChangeListener(this);
        thresholdSlider.setValue(sliderValue);
        thresholdSlider.addChangeListener(this);
        makeAndShowClusters();
      }
    }
    else
    if (e.getActionCommand().equals("run")) {
      double radius=Double.NaN;
      try {
        radius=Double.parseDouble(tfRadius.getText());
      } catch (Exception ex) {}
      if (Double.isNaN(radius) || radius<=0) {
        System.out.println("Illegal radius!");
        tfRadius.setText(String.format("%.5f",optics.getNeibRadius()));
        return;
      }
      int n=0;
      try {
        n=Integer.parseInt(tfNeighbors.getText());
      } catch (Exception ex) {}
      if (n<=0) {
        System.out.println("Illegal number of neighbours!");
        tfNeighbors.setText(Integer.toString(optics.getMinNeighbors()));
        return;
      }
      optics.doClustering(radius,n);
    }
  }
  
  public void makeAndShowClusters(){
    ClustersAssignments clAss=ClustererByOPTICS.makeClusters(objOrdered,rPlot.getThreshold());
    rPlot.setCusterAssignments(clAss);
    if (clAss!=null)
      labTop.setText(clAss.nClusters + " clusters; " + clAss.nNoise + " objects in noise; max cluster size = " +
                         clAss.maxSize + ", min cluster size = " + clAss.minSize);
    else
      labTop.setText("No clusters!");
    labTop.setSize(labTop.getPreferredSize());
    notifyChange(clAss);
  }
  
  protected ArrayList<ChangeListener> changeListeners=null;
  
  public void addChangeListener(ChangeListener l) {
    if (changeListeners==null)
      changeListeners=new ArrayList(5);
    if (!changeListeners.contains(l))
      changeListeners.add(l);
  }
  
  public void removeChangeListener(ChangeListener l) {
    if (l!=null && changeListeners!=null)
      changeListeners.remove(l);
  }
  
  public void notifyChange(ClustersAssignments clAss){
    if (changeListeners==null || changeListeners.isEmpty())
      return;
    ChangeEvent e=new ChangeEvent(clAss);
    for (ChangeListener l:changeListeners)
      l.stateChanged(e);
  }
}
