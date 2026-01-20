package TapasDataReader;

import java.util.TreeSet;

public class Flight {
  public String id;
  public int delays[]=null;
  public Explanation expl[]=null;
  public Flight (String id, TreeSet<Integer> steps) {
    this.id=id;
    delays=new int[steps.size()];
    for (int i=0; i<delays.length; i++)
      delays[i]=0;
  }
  public void createExpl() {
    expl=new Explanation[delays.length]; // same as steps.size(), see constructor
    for (int i=0; i<expl.length; i++)
      expl[i]=null;
  }
}
