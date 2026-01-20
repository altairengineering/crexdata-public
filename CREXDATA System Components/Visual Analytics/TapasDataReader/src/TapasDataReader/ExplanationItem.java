package TapasDataReader;

public class ExplanationItem {
  public int level;
  public String attr, // feature full name, e.g. DurationInSector02
         attr_core; // core of the attr name, e.g. DurationInSector
  public int attr_N; // N in the attr name, e.g. 2
  public String sector; // if present, otherwise null
  public float value;
  public double interval[]={Double.NEGATIVE_INFINITY,Double.POSITIVE_INFINITY}; // min..max; either min or max is +-inf
  public boolean isInteger=false;
  
  public ExplanationItem clone() {
    ExplanationItem ei=new ExplanationItem();
    ei.level=this.level;
    ei.value=this.value;
    ei.interval=this.interval.clone();
    ei.isInteger=isInteger;
    ei.attr=this.attr;
    ei.attr_core=this.attr_core;
    ei.attr_N=this.attr_N;
    ei.sector=this.sector;
    return ei;
  }
  
  public boolean equals(Object obj) {
    if (obj==null || !(obj instanceof ExplanationItem))
      return false;
    return equals((ExplanationItem)obj);
  }
  
  public boolean equals(ExplanationItem e) {
    if (e==null)
      return false;
    return sameCondition(e);
  }
  
  
  public boolean sameCondition(ExplanationItem ei) {
    if (ei.attr==null)
      return this.attr==null;
    if (!ei.attr.equals(this.attr))
      return false;
    if (ei.interval==null)
      return this.interval==null;
    if (this.interval==null)
      return false;
    return ei.interval[0]==this.interval[0] && ei.interval[1]==this.interval[1];
  }
}
