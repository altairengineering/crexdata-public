package TapasDataReader;

import java.awt.Color;
import java.util.AbstractList;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Hashtable;

/**
 * Represents a re-occurring explanation.
 */
public class CommonExplanation {
  /**
   * Numeric identifier of the rule
   */
  public int numId=-1;
  /**
   * Numeric idntifier of the upper rule in a hierarchy, if any
   */
  public int upperId=-1;
  /**
   * If rules originate from a decision forest (collection of trees), this is the
   * identifier of the tree from which the rule has been derived.
   * Additionally, if the trees were clustered, there is also the identifier of the tree cluster.
   */
  public int treeId=-1, treeCluster=-1;
  /**
   * Optional: any categorical attribute that must be shown in a table and can be used for filtering.
   */
  public String category=null;
  
  public ExplanationItem eItems[]=null;
  /**
   * The action (decision) explained by this explanation.
   */
  public int action=-1;
  /**
   * Number of duplicates of this explanation/rule (may occur in a decision forest)
   */
  public int nSame=1;
  /**
   * The "weight" of this rule, i.e., how many points it will have in voting among all rules applicable
   * to the same data instance. Initially, the weights of all rules equal 1. When to or more rules are
   * combined, the weight of the resulting rule equals is the sum of the weights of the original rules.
   */
  public int weight=1;
  /**
   * Individual explanations where the same attributes and conditions are used.
   * The individual explanations are grouped by the identifiers of the flights they refer to.
   * The keys of the hash table are the flight identifiers, the elements are the corresponding
   * individual explanations.
   */
  public Hashtable<String, ArrayList<Explanation>> uses=null;
  /**
   * Total number of uses of this explanation.
   */
  public int nUses=0;
  /**
   * Applications of this explanation or rule to data
   */
  public RuleApplication applications[]=null;
  /**
   * The numbers of right and wrong applications of the rule to data instances (called cases).
   */
  public int nCasesRight=0, nCasesWrong=0;
  /**
   * Q value, also used in regression rules / trees
   */
  public float minQ=Float.NaN, maxQ=Float.NaN, meanQ=Float.NaN;
  public double sumQ=Double.NaN;
  /**
   * Coordinate of this explanation in a 1D projection.
   */
  public double x1D=Double.NaN;
  /**
   * Coordinates of this explanation in a 2D or 3D projection (the length of the
   * array correspond to the number of dimensions)
   */
  public double coord[]=null;
  /**
   * A color assigned to this explanation.
   */
  public Color color=null;

  /**
   * Returns the value interval of the feature (attribute) with the given name or
   * null if no condition with this feature
   * @param attrName - feature name
   * @return value interval (array of length 2) or null
   */
  public double[] getFeatureInterval(String attrName) {
    if (attrName==null || eItems==null)
      return null;
    for (ExplanationItem ei:eItems)
      if (ei.attr.equals(attrName))
        return ei.interval;
    return null;
  }
  
  public boolean hasFeature(String attrName) {
    if (attrName==null || eItems==null)
      return false;
    for (ExplanationItem ei:eItems)
      if (ei.attr.equals(attrName))
        return true;
    return false;
  }
  
  public boolean equals(Object obj) {
    if (obj==null || !(obj instanceof CommonExplanation))
      return false;
    CommonExplanation cEx=(CommonExplanation)obj;
    if (this.action!=cEx.action)
      return false;
    if (this.eItems==null)
      return cEx.eItems==null;
    if (cEx.eItems==null)
      return false;
    if (this.eItems.length!=cEx.eItems.length)
      return false;
    for (int i=0; i<eItems.length; i++)
      if (!eItems[i].equals(cEx.eItems[i]))
        return false;
    return true;
  }

  public int getUsesCount() {
    return (uses!=null)?uses.size():nUses;
  }

  public int getApplicationsCount() {
    if (applications==null)
      return 0;
    return applications.length;
  }
  
  public String toString(){
    String str=((Double.isNaN(meanQ))?"Class = "+action:
        (minQ<maxQ)?String.format("Range = [%.3f,%.3f]",minQ,maxQ):
            String.format("Value = %.3f",meanQ))+
        "; applied "+nUses+" times";
    if (nCasesRight>0 || nCasesWrong>0)
      str+="; right:wrong = "+nCasesRight+":"+nCasesWrong;
    for (int i=0; i<eItems.length; i++) {
      str +="; "+eItems[i].attr;
      if (Double.isInfinite(eItems[i].interval[0]))
        str+="<"+eItems[i].interval[1];
      else
        if (Double.isInfinite(eItems[i].interval[1]))
          str+=">"+eItems[i].interval[0];
        else
          str+=" in ["+eItems[i].interval[0]+".."+eItems[i].interval[1]+")";
          
    }
    return str;
  }

  public String toHTML (ArrayList<String> listOfFeatures, Hashtable<String,float[]> attrMinMax) {
    return toHTML(listOfFeatures,attrMinMax,"",null);
  }

  public String toHTML (ArrayList<String> listOfFeatures, Hashtable<String,float[]> attrMinMax,
                        String columnAtPointer, String imgFile) {
    //System.out.println(columnAtPointer);
    String txt="<html><body style=background-color:rgb(255,255,204)>";
    txt+="<p style=\"text-align:center;\">";
    if (numId>=0)
      if (upperId>=0)
        txt+="Rule <b>"+numId+"</b>; upper rule <b>"+upperId+"</b>";
      else
        txt+="Rule <b>"+numId+"</b>";
    txt+="&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Weight = "+weight+"</p>";
    txt += "<table border=0 cellmargin=3 cellpadding=3 cellspacing=3 align=center>";
    txt+="<tr align=right><td>Class </td><td><b>"+action+"</b></td>";
    if (!Float.isNaN(meanQ))
      txt+="<td>Mean Q</td><td>"+String.format("%.4f",meanQ)+"</td>";
    txt+="</tr>";
    if (!Float.isNaN(minQ))
      txt+="<tr align=right><td>N uses:</td><td>"+nUses+"</td><td>Min Q</td><td>"+String.format("%.4f",minQ)+"</td></tr>";
    if (!Float.isNaN(maxQ))
      txt+="<tr align=right><td>N distinct items:</td><td>"+getUsesCount()+"</td><td>Max Q</td><td>"+String.format("%.4f",maxQ)+"</td></tr>";
    txt += "<tr></tr></table>";
    if (imgFile!=null)
      txt+="<p align=center><img border=1 src=file:"+imgFile+" width=100%></p>";
    txt += "<table border=1 cellmargin=3 cellpadding=3 cellspacing=3>";

    txt+="<tr><td>Feature</td><td>min</td><td>from</td><td>to</td><td>max</td></tr>";
    if (listOfFeatures==null)
      for (int i=0; i<eItems.length; i++)
        txt+=processEItem(i,attrMinMax,columnAtPointer,imgFile);
    else
      for (int idx=0; idx<listOfFeatures.size(); idx++)
        for (int i=0; i<eItems.length; i++)
          if (eItems[i].attr.equals(listOfFeatures.get(idx)))
            txt+=processEItem(i,attrMinMax,columnAtPointer,imgFile);
    txt += "</table>";
    txt+="</body></html>";
    return txt;
  }

  protected String processEItem (int i, Hashtable<String,float[]> attrMinMax, String columnAtPointer, String imgFile) {
    String txt="";
    boolean b=eItems[i].attr.equals(columnAtPointer);
    txt+="<tr align=right><td>"+((b)?"<b>":"")+eItems[i].attr+((b)?"</b>":"")+"</td>";
    String strValue=(attrMinMax!=null && attrMinMax.get(eItems[i].attr)!=null)?
            String.format("%.4f",attrMinMax.get(eItems[i].attr)[0]):"";
    txt+="<td>"+strValue+"</td><td>";
    if (!Double.isInfinite(eItems[i].interval[0]))
      txt+=(eItems[i].isInteger)?String.valueOf((int)eItems[i].interval[0]):String.format("%.4f",eItems[i].interval[0]);
    else
      txt+="- inf";
    txt+="</td><td>";
    if (!Double.isInfinite(eItems[i].interval[1]))
      txt+=(eItems[i].isInteger)?String.valueOf((int)eItems[i].interval[1]):String.format("%.4f",eItems[i].interval[1]);
    else
      txt+="+ inf";
    txt+="</td>";
    strValue=(attrMinMax!=null && attrMinMax.get(eItems[i].attr)!=null)?
            String.format("%.4f",attrMinMax.get(eItems[i].attr)[1]):"";
    txt+="<td>"+strValue+"</td></tr>";
    return txt;
  }
  
  /**
   * If both explanations refer to the same action, returns true if this explanation is
   * more general than ex (subsumes ex).
   */
  public boolean subsumes(CommonExplanation ex) {
    return subsumes(ex,true);
  }
  /**
   * If both explanations refer to the same action, returns true if this explanation is
   * more general than ex (subsumes ex).
   */
  public boolean subsumes(CommonExplanation ex, boolean mustHaveSameAction) {
    if (ex==null)
      return false;
    if (mustHaveSameAction && ex.action!=this.action)
      return false;
    ExplanationItem e1[]=eItems, e2[]=ex.eItems;
    if (e1==null || e1.length<1 || e2==null || e2.length<1)
      return false;
    boolean subsumes=true;
    for (int i=0; i<e1.length && subsumes; i++) {
      int i2 = -1;
      for (int j = 0; j < e2.length && i2 < 0; j++)
        if (e1[i].attr.equals(e2[j].attr))
          i2 = j;
      subsumes = i2>=0 && includes(e1[i].interval,e2[i2].interval);
    }
    return subsumes;
  }
  
  /**
   * Counts the numbers of right and wrong applications of the explanation or rule to data instances.
   * @param exData  - data instances
   */
  public void countRightAndWrongApplications(AbstractList<Explanation> exData, boolean byAction) {
    nCasesRight=nCasesWrong=0;
    if (exData==null || exData.isEmpty() || eItems==null)
      return;
    
    ArrayList<RuleApplication> appList=new ArrayList<RuleApplication>(100);
    
    for (Explanation ex:exData) {
      boolean ruleApplies=true;
      for (int i=0; i<eItems.length && ruleApplies; i++) {
        double range[]=eItems[i].interval, value=Double.NaN;
        for (int j=0; j<ex.eItems.length && Double.isNaN(value); j++)
          if (eItems[i].attr.equals(ex.eItems[j].attr)) {
            value=ex.eItems[j].value;
            break;
          }
        ruleApplies=!Double.isNaN(value) &&
                        (Double.isNaN(range[0]) || Double.isInfinite(range[0]) || value>=range[0]) &&
                        (Double.isNaN(range[1]) || Double.isInfinite(range[1]) || value<=range[1]);
      }
      if (!ruleApplies)
        continue;
      
      RuleApplication app=new RuleApplication();
      app.data=ex;
      if (byAction)
        app.isRight= this.action==ex.action;
      else
        app.isRight= ex.Q>=this.minQ && ex.Q<=this.maxQ;
      appList.add(app);
      
      if (app.isRight)
          ++nCasesRight;
        else
          ++nCasesWrong;
    }
    
    applications=(appList==null)?null:appList.toArray(new RuleApplication[appList.size()]);
  }
  
  public static boolean includes(double interval1[], double interval2[]) {
    if (interval1==null || interval2==null)
      return false;
    return interval1[0]<=interval2[0] && interval1[1]>=interval2[1];
  }
  
  /**
   * Creates a new instance of CommonExplanation from the given individual explanations.
   * Puts the individual explanation in the hash table "uses".
   */
  public static CommonExplanation createCommonExplanation(Explanation ex,
                                                          boolean transformConditionsToInteger,
                                                          Hashtable<String,float[]> attrMinMax,
                                                          boolean combineConditions) {
    if (ex==null || ex.eItems==null || ex.eItems.length<1)
      return null;
    ExplanationItem ei[]=makeCopyAndTransform(ex.eItems,
        transformConditionsToInteger,attrMinMax,combineConditions);
    
    CommonExplanation cEx=new CommonExplanation();
    cEx.eItems=ei;
    cEx.action=ex.action;
    cEx.uses=new Hashtable<String, ArrayList<Explanation>>(25);
    ArrayList<Explanation> flightExpl=new ArrayList<Explanation>(10);
    flightExpl.add(ex);
    cEx.uses.put(ex.FlightID,flightExpl);
    cEx.nUses=1;
    if (!Float.isNaN(ex.Q)) {
      cEx.minQ = cEx.maxQ = cEx.meanQ = ex.Q;
      cEx.sumQ=ex.Q;
    }
    return cEx;
  }
  
  public static ExplanationItem[] makeCopy(ExplanationItem ei[]){
    if (ei==null || ei.length==0)
      return null;
    ExplanationItem eii[]=new ExplanationItem[ei.length];
    for (int i=0; i<ei.length; i++)
      eii[i]=ei[i].clone();
    return eii;
  }
  
  public static ExplanationItem[] makeCopyAndTransform(ExplanationItem eiOrig[],
                                                       boolean transformConditionsToInteger,
                                                       Hashtable<String,float[]> attrMinMax,
                                                       boolean combineConditions){
    ExplanationItem ei[]=makeCopy(eiOrig);
    if (ei==null)
      return null;
    if (transformConditionsToInteger)
      for (int i=0; i<ei.length; i++) {
        ei[i].isInteger=true;
        double a=ei[i].interval[0], b=ei[i].interval[1];
        if (!Double.isInfinite(a) && !Double.isInfinite(b) &&
                Math.floor(a)==Math.ceil(a) && Math.floor(b)==Math.ceil(b))
          continue; //both are already integers
        float minmax[]=(attrMinMax==null)?null:attrMinMax.get(ei[i].attr);
        if (minmax!=null) {
          ei[i].interval[0]=Math.max(minmax[0],Math.ceil(a));
          ei[i].interval[1]=Math.min(minmax[1],Math.floor(b));
        }
        else {
          ei[i].interval[0]=Math.ceil(a);
          ei[i].interval[1]=Math.floor(b);
        }
      }
    if (combineConditions)
      ei=Explanation.getExplItemsCombined(ei);
    return ei;
  }
  
  public static boolean sameFeatures(ExplanationItem e1[], ExplanationItem e2[]) {
    if (e1==null || e1.length<1)
      return e2==null || e2.length<1;
    boolean ok[]=new boolean[e2.length];
    for (int i=0; i<ok.length; i++)
      ok[i]=false;
    for (int i=0; i<e1.length; i++) {
      boolean found=false;
      for (int j=0; j<e2.length && !found; j++) {
        found = e1[i].attr.equals(e2[j].attr);
        if (found)
          ok[j]=true;
      }
      if (!found)
        return false;
    }
    for (int i=0; i<e2.length; i++)
      if (!ok[i]) {
        boolean found=false;
        for (int j=0; j<e1.length && !found; j++)
          found = e2[i].attr.equals(e1[j].attr);
        if (!found)
          return false;
      }
    return true;
  }
  
  public static boolean sameFeatures(CommonExplanation ex1, CommonExplanation ex2) {
    if (ex1==null || ex2==null)
      return false;
    return sameFeatures(ex1.eItems,ex2.eItems);
  }
  
  public static boolean sameExplanations(ExplanationItem e1[], ExplanationItem e2[]) {
    if (e1==null || e1.length<1)
      return e2==null || e2.length<1;
    boolean ok[]=new boolean[e2.length];
    for (int i=0; i<ok.length; i++)
      ok[i]=false;
    for (int i=0; i<e1.length; i++) {
      boolean found=false;
      for (int j=0; j<e2.length && !found; j++) {
        found = e1[i].sameCondition(e2[j]);
        if (found)
          ok[j]=true;
      }
      if (!found)
        return false;
    }
    for (int i=0; i<e2.length; i++)
      if (!ok[i]) {
        boolean found=false;
        for (int j=0; j<e1.length && !found; j++)
          found = e2[i].sameCondition(e1[j]);
        if (!found)
          return false;
      }
    return true;
  }
  
  public static boolean sameExplanations(CommonExplanation ex1, CommonExplanation ex2) {
    if (ex1==null || ex2==null)
      return false;
    return sameExplanations(ex1.eItems,ex2.eItems);
  }
  
  public static ArrayList<CommonExplanation> addExplanation(ArrayList<CommonExplanation> exList,
                                                            Explanation exToAdd,
                                                            boolean transformConditionsToInteger,
                                                            Hashtable<String,float[]> attrMinMax,
                                                            boolean combineConditions) {
    if (exToAdd==null || exToAdd.eItems==null || exToAdd.eItems.length<1)
      return null;
    ExplanationItem ei[]=makeCopyAndTransform(exToAdd.eItems,
        transformConditionsToInteger,attrMinMax,combineConditions);
    if (exList==null)
      exList=new ArrayList<CommonExplanation>(1000);
    CommonExplanation cEx=null;
    for (int i=0; i<exList.size() && cEx==null; i++) {
      cEx=exList.get(i);
      if (exToAdd.action!=cEx.action || !sameExplanations(ei,cEx.eItems))
        cEx=null;
    }
    if (cEx==null) {
      cEx=new CommonExplanation();
      cEx.eItems=ei;
      cEx.action=exToAdd.action;
      cEx.uses=new Hashtable<String, ArrayList<Explanation>>(25);
      exList.add(cEx);
    }
    ArrayList<Explanation> flightExpl=cEx.uses.get(exToAdd.FlightID);
    if (flightExpl==null)
      flightExpl=new ArrayList<Explanation>(10);
    flightExpl.add(exToAdd);
    cEx.uses.put(exToAdd.FlightID,flightExpl);
    ++cEx.nUses;
    if (!Float.isNaN(exToAdd.Q)) {
      cEx.sumQ+=exToAdd.Q;
      cEx.meanQ=(float)cEx.sumQ/cEx.nUses;
      if (Float.isNaN(cEx.minQ) || cEx.minQ>exToAdd.Q)
        cEx.minQ=exToAdd.Q;
      if (Float.isNaN(cEx.maxQ) || cEx.maxQ<exToAdd.Q)
        cEx.maxQ=exToAdd.Q;
    }
    return exList;
  }
  
  public static ArrayList<CommonExplanation> getCommonExplanations(ArrayList<Explanation> explanations,
                                                                   boolean transformConditionsToInteger,
                                                                   Hashtable<String,float[]> attrMinMax,
                                                                   boolean combineConditions) {
    if (explanations==null || explanations.isEmpty())
      return null;
    ArrayList<CommonExplanation> exList=null;
    for (int e=0; e<explanations.size(); e++) {
      Explanation ex = explanations.get(e);
      if (ex == null || ex.eItems == null)
        continue;
      exList=addExplanation(exList,ex,transformConditionsToInteger,attrMinMax,combineConditions);
    }
    return exList;
  }
  
  public static double[][] computeDistances(ArrayList<CommonExplanation> explanations,
                                            Hashtable<String,float[]> attrMinMaxValues) {
    if (explanations==null || explanations.size()<2)
      return null;
    double d[][]=new double[explanations.size()][explanations.size()];
    for (int i=0; i<d.length; i++) {
      d[i][i]=0;
      for (int j=i+1; j<d.length; j++)
        d[i][j]=d[j][i]=distance(explanations.get(i).eItems,explanations.get(j).eItems,attrMinMaxValues);
    }
    return d;
  }

  /*
  public static double distance(ExplanationItem e1[], ExplanationItem e2[],
                                Hashtable<String,float[]> attrMinMaxValues) {
    if (e1==null || e1.length<1)
      if (e2==null) return 0; else return e2.length;
    if (e2==null || e2.length<1)
      return e1.length;
    double d=0;
    boolean e2InE1[]=new boolean[e2.length];
    for (int i=0; i<e2.length; i++)
      e2InE1[i]=false;
    int nCommon=0;
    for (int i=0; i<e1.length; i++) {
      double inter1[]=e1[i].interval;
      float minmax[]=(attrMinMaxValues==null)?null:attrMinMaxValues.get(e1[i].attr);
      double dMinMax[]={(minmax==null)?Double.NEGATIVE_INFINITY:minmax[0],
          (minmax==null)?Double.POSITIVE_INFINITY:minmax[1]};
      int i2=-1;
      for (int j=0; j<e2.length && i2<0; j++)
        if (e1[i].attr.equals(e2[j].attr))
          i2=j;
      if (i2>=0) {
        e2InE1[i2]=true;
        ++nCommon;
      }
      double inter2[]=(i2>=0)?e2[i2].interval:dMinMax;
      d+=IntervalDistance.distanceRelative(inter1[0],inter1[1],
          inter2[0],inter2[1],dMinMax[0],dMinMax[1]);
    }
    if (nCommon<e2.length)
      for (int i=0; i<e2.length; i++)
        if (!e2InE1[i]) {
          float minmax[]=(attrMinMaxValues==null)?null:attrMinMaxValues.get(e2[i].attr);
          double dMinMax[]={(minmax==null)?Double.NEGATIVE_INFINITY:minmax[0],
              (minmax==null)?Double.POSITIVE_INFINITY:minmax[1]};
          d+=IntervalDistance.distanceRelative(e2[i].interval[0],e2[i].interval[1],
              dMinMax[0],dMinMax[1],dMinMax[0],dMinMax[1]);
        }
    return d;
  }
  */

  public static double[][] computeDistances(ArrayList<CommonExplanation> explanations,
                                            HashSet<String> featuresToUse,
                                            Hashtable<String,float[]> attrMinMaxValues) {
    if (explanations==null || explanations.size()<2)
      return null;
    if (featuresToUse==null || featuresToUse.isEmpty())
      return computeDistances(explanations,attrMinMaxValues);
    double d[][]=new double[explanations.size()][explanations.size()];
    for (int i=0; i<d.length; i++) {
      d[i][i]=0;
      for (int j=i+1; j<d.length; j++)
        d[i][j]=d[j][i]=distance(explanations.get(i).eItems,
            explanations.get(j).eItems,featuresToUse,attrMinMaxValues);
    }
    return d;
  }

  public static double distance(ExplanationItem e1[], ExplanationItem e2[],
                                Hashtable<String,float[]> attrMinMaxValues) {
    if (e1==null && e2==null)
      return 0;
    HashSet<String> features=new HashSet<String>(50);
    if (e1!=null)
      for (ExplanationItem e:e1)
        if (!features.contains(e.attr))
          features.add(e.attr);
    if (e2!=null)
      for (ExplanationItem e:e2)
        if (!features.contains(e.attr))
          features.add(e.attr);
    return distance(e1,e2,features,attrMinMaxValues);
  }


  public static double distance(ExplanationItem e1[], ExplanationItem e2[],
                                HashSet<String> featuresToUse,
                                Hashtable<String,float[]> attrMinMaxValues) {
    if (e1==null && e2==null)
      return 0;
    if (featuresToUse==null || featuresToUse.isEmpty())
      return distance(e1,e2,attrMinMaxValues);
    double d=0;
    HashSet<String> featuresChecked=new HashSet<String>(featuresToUse.size());
    for (String attr:featuresToUse) {
      int i1=-1, i2=-1;
      if (e1!=null)
        for (int i=0; i<e1.length && i1<0; i++)
          if (attr.equals(e1[i].attr))
            i1=i;
      if (e2!=null)
        for (int i=0; i<e2.length && i2<0; i++)
          if (attr.equals(e2[i].attr))
            i2=i;
      if (i1<0 && i2<0)
        continue;
      float minmax[]=(attrMinMaxValues==null)?null:attrMinMaxValues.get(attr);
      double dMinMax[]={(minmax==null)?Double.NEGATIVE_INFINITY:minmax[0],
          (minmax==null)?Double.POSITIVE_INFINITY:minmax[1]};
      double inter1[]=(i1>=0)?e1[i1].interval:dMinMax;
      double inter2[]=(i2>=0)?e2[i2].interval:dMinMax;
      d+=IntervalDistance.distanceRelative(inter1[0],inter1[1],
          inter2[0],inter2[1],dMinMax[0],dMinMax[1]);
    }
    return d;
  }

  /*
  public static double distance(ExplanationItem e1[], ExplanationItem e2[],
                                Hashtable<String,float[]> attrMinMaxValues) {
    if (e1==null || e1.length<1)
      if (e2==null) return 0; else return e2.length;
    if (e2==null || e2.length<1)
      return e1.length;
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
  */
}
