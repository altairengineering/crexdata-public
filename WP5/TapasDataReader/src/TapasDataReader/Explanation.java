package TapasDataReader;

import java.util.Hashtable;
import java.util.Vector;

/**
 * an array of Explanations for all steps is attached to each flight.
 * to save memory, step and flight id are not recorder here (to be deleted from here after debugging)
 */
public class Explanation {
  public String FlightID=null;
  public int step=-1;
  public int action=-1;
  public float Q=Float.NaN;
  public ExplanationItem eItems[]=null;

  public static ExplanationItem[] getExplItemsCombined (ExplanationItem ei[]) {
    if (ei==null || ei.length==0)
      return null;
    Vector<ExplanationItem> vei=new Vector<ExplanationItem>(ei.length);
    vei.add(ei[0].clone());
    for (int i=1; i<ei.length; i++)
    if (ei[i]==null)
      ; //System.out.println("null eItem, flight "+FlightID+", step = "+step);
    else {
      int n=-1;
      for (int j=0; n==-1 && j<vei.size(); j++)
        if (vei.elementAt(j).attr.equals(ei[i].attr))
          n=j;
      if (n==-1) // add Ith condition to the explanation
        vei.add(ei[i].clone());
      else { // combine Nth and Ith conditions
        ExplanationItem e=vei.elementAt(n);
        e.interval[0]=Math.max(e.interval[0],ei[i].interval[0]);
        e.interval[1]=Math.min(e.interval[1],ei[i].interval[1]);
      }
    }
    ExplanationItem eItemsCombined[]=new ExplanationItem[vei.size()];
    for (int i=0; i<vei.size(); i++)
      eItemsCombined[i]=vei.elementAt(i);
    return eItemsCombined;
  }

  public static ExplanationItem[] getExplItemsAsIntegeres (ExplanationItem ei[], Hashtable<String,float[]> attrs) {
    if (ei==null || ei.length==0)
      return null;
    ExplanationItem eii[]=new ExplanationItem[ei.length];
    for (int i=0; i<ei.length; i++) {
      eii[i]=ei[i].clone();
      float minmax[]=attrs.get(ei[i].attr);
      eii[i].interval[0]=Math.max(minmax[0],Math.ceil(ei[i].interval[0]));
      eii[i].interval[1]=Math.min(minmax[1],Math.floor(ei[i].interval[1]));
      eii[i].isInteger=true;
    }
    return eii;
  }
  
  public static double distance(ExplanationItem e1[], ExplanationItem e2[],
                                Hashtable<String,float[]> attrMinMaxValues) {
    if (e1==null || e1.length<1)
      if (e2==null) return 0; else return e2.length;
    if (e2==null || e2.length<1)
      return e1.length;
    e1=getExplItemsCombined(e1);
    e2=getExplItemsCombined(e2);
    double d=e1.length+e2.length;
    for (int i=0; i<e1.length; i++) {
      int i2=-1;
      for (int j=0; j<e2.length && i2<0; j++)
        if (e1[i].attr.equals(e2[j].attr))
          i2=j;
      if (i2<0)
        continue;
      d-=2; //corresponding items found
      float minmax[]=attrMinMaxValues.get(e1[i].attr);
      double min=(minmax==null)?Double.NaN:minmax[0], max=(minmax==null)?Double.NaN:minmax[1];
      d+=IntervalDistance.distanceRelative(e1[i].interval[0],e1[i].interval[1],
          e2[i2].interval[0],e2[i2].interval[1],min,max);
    }
    return d;
  }
}
