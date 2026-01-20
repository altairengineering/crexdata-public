package TapasUtilities;

import java.util.ArrayList;
import java.util.Hashtable;

/**
 * Establishes correspondence between two sets of objects.
 * Can be used, e.g., for synchronous highlighting or selection.
 */
public class Translator {
  /**
   * An array of pairs specifying correspondence between objects of two sets.
   */
  public ArrayList<Object[]> pairs=null;
  /**
   * Indexes of the objects of the first and second set in the list of pairs.
   */
  public Hashtable<Object,Integer> indexes1=null, indexes2=null;
  
  public void setPairs(ArrayList<Object[]> pairs) {
    this.pairs=pairs;
    if (pairs==null || pairs.isEmpty())
      return;
    indexes1=new Hashtable<Object, Integer>(pairs.size());
    indexes2=new Hashtable<Object, Integer>(pairs.size());
    for (int i=0; i<pairs.size(); i++) {
      Object pair[]=pairs.get(i);
      indexes1.put(pair[0],i);
      indexes2.put(pair[1],i);
    }
  }
  
  public Object getSecond(Object first) {
    if (indexes1!=null) {
      Integer idx=indexes1.get(first);
      if (idx!=null)
        return pairs.get(idx)[1];
    }
    return null;
  }
  
  public Object getFirst(Object second) {
    if (indexes2!=null) {
      Integer idx=indexes2.get(second);
      if (idx!=null)
        return pairs.get(idx)[0];
    }
    return null;
  }
}
