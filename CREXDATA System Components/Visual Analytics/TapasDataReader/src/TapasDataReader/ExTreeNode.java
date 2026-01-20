package TapasDataReader;

import java.util.ArrayList;
import java.util.Hashtable;
import java.util.TreeSet;

/**
 * Represents a node in an explanation tree (decision tree)
 */
public class ExTreeNode {
  public int level=0;
  public String attrName=null;
  /**
   * min..max; either min or max is +-inf
   */
  public double condition[]={Double.NEGATIVE_INFINITY,Double.POSITIVE_INFINITY};
  public boolean isInteger=false;
  /**
   * How many times this node was used (i.e., occurs in explanations)
   */
  public int nUses=0;
  /**
   * The list of solution steps in which this node was used.
   * Such a list may be created when an explanation tree for a single flight is reconstructed.
   * In other cases, it be null.
   */
  public TreeSet<Integer> steps=null;
  /**
   * How many times different sectors appear in combination with this attribute and this condition
   */
  public Hashtable<String,Integer> sectors=null;
  
  public ExTreeNode parent=null;
  public ArrayList<ExTreeNode> children=null;
  /**
   * How many terminal nodes (leaves) exist in the subtree originating from this node.
   * If this is a leaf, the value is 0.
   */
  public int nLeavesBelow=0;
  
  public String getLabel() {
    if (attrName==null)
      return "null";
    if (condition==null ||
            (condition[0]==Double.NEGATIVE_INFINITY &&
                 condition[1]==Double.POSITIVE_INFINITY))
      return attrName;
    String c1=(isInteger)?String.valueOf(Math.round(condition[0])):String.valueOf(condition[0]);
    String c2=(isInteger)?String.valueOf(Math.round(condition[1])):String.valueOf(condition[1]);
    if (condition[0]==condition[1])
      return attrName+" = "+c1;
    if (condition[0]==Double.NEGATIVE_INFINITY)
      return attrName+" < "+c2;
    if (condition[1]==Double.POSITIVE_INFINITY)
      return attrName+" >= "+c1;
    return attrName+" in ["+c1+", "+c2+((isInteger)?"]":")");
  }
  
  public boolean sameCondition(String attrName, double condition[]) {
    if (attrName==null)
      return this.attrName==null;
    if (!attrName.equals(this.attrName))
      return false;
    if (condition==null)
      return this.condition==null;
    if (this.condition==null)
      return false;
    return condition[0]==this.condition[0] && condition[1]==this.condition[1];
  }
  
  static public int compareConditions(double cnd1[], double cnd2[]) {
    if (cnd1==null)
      return (cnd2==null)?0:1;
    if (cnd2==null) return -1;
    if (cnd1[0]==cnd2[0])
      return (cnd1[1]<cnd2[1])?-1:(cnd1[1]==cnd2[1])?0:1;
    return (cnd1[0]<cnd2[1])?-1:1;
  }
  
  public ExTreeNode findChild(String attrName, double condition[]) {
    if (children==null)
      return null;
    for (ExTreeNode child:children)
      if (child.sameCondition(attrName,condition))
        return child;
    return null;
  }
  
  public void addUse(){
    ++nUses;
  }
  
  public void addSectorUse(String sector){
    if (sectors==null)
      sectors=new Hashtable<String,Integer>(100);
    Integer n=sectors.get(sector);
    if (n==null) n=0;
    sectors.put(sector,n+1);
  }
  
  public void addStep(int step) {
    if (steps==null)
      steps=new TreeSet<Integer>();
    steps.add(step);
  }
  
  public void addChild(ExTreeNode child) {
    if (child==null)
      return;
    if (children==null)
      children=new ArrayList<ExTreeNode>(10);
    int idx=-1;
    if (!children.isEmpty()) {
      for (int i = 0; i < children.size() && idx < 0; i++)
        if (child.attrName.equals(children.get(i).attrName))
          idx = (compareConditions(child.condition, children.get(i).condition) < 0) ? i : i + 1;
      if (idx<0)
        for (int i = 0; i < children.size() && idx < 0; i++)
          if (child.attrName.compareTo(children.get(i).attrName)<0)
            idx=i;
    }
    if (idx>=0)
      children.add(idx,child);
    else
      children.add(child);
    child.parent=this;
  }
  
  public void countLeavesBelow() {
    nLeavesBelow=0;
    if (children!=null)
      for (ExTreeNode child:children) {
        child.countLeavesBelow();
        nLeavesBelow+=Math.max(1,child.nLeavesBelow);
      }
  }
}
