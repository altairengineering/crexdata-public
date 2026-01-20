package TapasExplTreeViewer.vis;

import TapasUtilities.MySammonsProjection;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.util.ArrayList;

public class SammonsMappingRunner implements ChangeListener, ProjectionProvider {
  /**
   * Matrix of distances between the objects to project and show
   */
  public double distances[][]=null;
  /**
   * Creates a projection based on the distance matrix
   */
  protected MySammonsProjection sam=null;
  /**
   * The projection obtained (updated iteratively)
   */
  protected double proj[][]=null;
  
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
  
  public void notifyChange(){
    if (changeListeners==null || changeListeners.isEmpty())
      return;
    ChangeEvent e=new ChangeEvent(this);
    for (ChangeListener l:changeListeners)
      l.stateChanged(e);
  }
  
  public void setDistanceMatrix(double distances[][]) {
    this.distances=distances;
    if (distances!=null) {
      ChangeListener listener=this;
      SwingWorker worker=new SwingWorker() {
        @Override
        public Boolean doInBackground(){
          sam=new MySammonsProjection(distances,2,250,true);
          sam.runProjection(5,50,listener);
          return true;
        }
        @Override
        protected void done() {
        }
      };
      worker.execute();
    }
  }
  
  public double[][] getProjection(){
    return proj;
  }
  
  public void stateChanged(ChangeEvent e) {
    if (e.getSource() instanceof MySammonsProjection) {
      MySammonsProjection sam = (MySammonsProjection) e.getSource();
      proj = (sam.done) ? sam.getProjection() : sam.bestProjection;
      notifyChange();
    }
  }
  
  public String getProjectionTitle() {
    return "Sammon\'s mapping";
  }
}
