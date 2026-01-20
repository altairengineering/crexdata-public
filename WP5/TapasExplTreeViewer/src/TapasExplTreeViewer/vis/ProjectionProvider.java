package TapasExplTreeViewer.vis;

import javax.swing.event.ChangeListener;

public interface ProjectionProvider {
  public void setDistanceMatrix(double distances[][]);
  public double[][] getProjection();
  public void addChangeListener(ChangeListener l);
  public String getProjectionTitle();
}
