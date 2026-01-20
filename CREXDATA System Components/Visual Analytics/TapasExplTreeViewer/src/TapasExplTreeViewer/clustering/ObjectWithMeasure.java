package TapasExplTreeViewer.clustering;

/**
 * Created by IntelliJ IDEA.
 * Creator: N.Andrienko
 * Date: 24-Apr-2007
 * Time: 10:15:50
 * Used for sorting objects according to some numeric measures attached to them.
 */
public class ObjectWithMeasure implements Comparable<ObjectWithMeasure> {
  public Object obj=null;
  public double measure=Double.NaN;
  public boolean decreasingOrder=false;

  public ObjectWithMeasure() {}

  public ObjectWithMeasure(Object object, double measure) {
    obj=object; this.measure=measure;
  }

  public ObjectWithMeasure(Object object, double measure, boolean decreasingOrder) {
    obj=object; this.measure=measure; this.decreasingOrder=decreasingOrder;
  }
  /**
  *  Returns 0 if equal, <0 if THIS is less than the argument, >0 otherwise
  */
  public int compareTo (Comparable c) {
    if (c==null || !(c instanceof ObjectWithMeasure)) return -1;
    return compareTo((ObjectWithMeasure)c);
  }
  /**
   *  Returns 0 if equal, <0 if THIS is less than the argument, >0 otherwise
   */
  public int compareTo (ObjectWithMeasure om) {
    if (Double.isNaN(measure))
      if (Double.isNaN(om.measure)) return 0;
      else return 1;
    if (Double.isNaN(om.measure)) return -1;
    if (measure<om.measure)
      return (decreasingOrder)?1:-1;
    if (measure>om.measure)
      return (decreasingOrder)?-1:1;
    return 0;
  }

  public String toString() {
    return obj+":"+measure;
  }
}
