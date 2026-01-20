package TapasExplTreeViewer.rules;

import TapasDataReader.CommonExplanation;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Hashtable;
import java.util.Map;

public class RuleSet {
  public String versionLabel="R0";
  public String title=null, description=null;

  public ArrayList<CommonExplanation> rules=null;

  /**
   * For classification rules: minimal and maximal index of the predicted class, action, or decision option.
   */
  public int minAction=Integer.MAX_VALUE, maxAction=Integer.MIN_VALUE;
  /**
   * For regression rules: minimum and maximum of the predicted numeric value
   */
  public double minQValue=Double.NaN, maxQValue=Double.NaN;
  /**
   * Whether there are different actions (decisions, classes) in the data and in the original rules
   */
  public boolean actionsDiffer =false;

  /**
   * list of features sorted by frequency
   */
  public ArrayList<String> listOfFeatures=null;
  /**
   * Pre-ordered list of feature names, e.g., according to user's preferences
   */
  public ArrayList<String> orderedFeatureNames=null;
  /**
   * The ranges of feature values
   */
  public Hashtable<String,float[]> attrMinMax=null;
  /**
   * Whether the features should be treated as binary
   */
  public boolean featuresAreBinary=false;
  /**
   * Whether the rule set has been previously reduced by removing rules
   * subsumed in more general rules.
   */
  public boolean nonSubsumed =false;
  /**
   * Whether the rule set consists of generalized rules obtained by aggregation.
   */
  public boolean aggregated=false;
  /**
   * The accuracy threshold used in aggregation
   */
  public double accThreshold=1;
  /**
   * The threshold for the differences in Q used for the aggregation
   */
  public double maxQDiff=0;
  /**
   * Whether the rule set consists of expanded rule hierarchies
   */
  public boolean expanded=false;

  /**
   * Which features are to be used in computing distances between rules
   */
  public HashSet featuresInDistances=null;
  /**
   * The distances between the rules
   */
  public double distanceMatrix[][]=null;

  public RuleSet parent=null;
  public ArrayList<RuleSet> children=null;
  
  public boolean hasRules() {
    return rules!=null && !rules.isEmpty();
  }

  public void determinePredictionRanges() {
    if (!hasRules())
      return;
    if (minAction<=maxAction || (!Double.isNaN(minQValue) && !Double.isNaN(maxQValue)))
      return; //was already done

    minAction=Integer.MAX_VALUE; maxAction=Integer.MIN_VALUE;
    minQValue=Double.NaN; maxQValue=Double.NaN;
    for (CommonExplanation ex:rules) {
      if (minAction > ex.action)
        minAction = ex.action;
      if (maxAction < ex.action)
        maxAction = ex.action;
      if (!Double.isNaN(ex.meanQ)) {
        if (Double.isNaN(minQValue) || minQValue > ex.minQ)
          minQValue = ex.minQ;
        if (Double.isNaN(maxQValue) || maxQValue < ex.maxQ)
          maxQValue = ex.maxQ;
      }
    }
    actionsDiffer=minAction<maxAction;
  }

  public int[] getMinMaxClass() {
    if (minAction>maxAction)
      determinePredictionRanges();
    int minmax[]={minAction,maxAction};
    return minmax;
  }

  public double[] getMinMaxQValue() {
    if (Double.isNaN(minQValue) && Double.isNaN(maxQValue))
      determinePredictionRanges();
    double minmax[]={minQValue,maxQValue};
    return minmax;
  }

  public boolean isRegression() {
    return !Double.isNaN(minQValue) && !Double.isNaN(maxQValue) && minQValue<maxQValue;
  }

  public ArrayList<CommonExplanation> selectRulesByConditionFilters(Map<String,Object> filters,
                                                                    boolean rangesMustBeInsideLimits) {
    if (rules==null || rules.isEmpty())
      return null;
    if (filters==null || filters.isEmpty())
      return rules;
    ArrayList<CommonExplanation> selected=new ArrayList<CommonExplanation>(rules.size()/2);
    for (CommonExplanation rule:rules) {
      boolean ok=true;
      for (String feature:filters.keySet()) {
        double range[]=rule.getFeatureInterval(feature);
        Object filter=filters.get(feature);
        double limits[]=(filter instanceof double[])?(double[]) filter:null;

        if (rangesMustBeInsideLimits)
          ok=doesLimitsIncludeRange(range,limits);
        else
          ok=doIntervalsOverlap(range,limits);

        if (!ok)
          break;
      }
      if (ok)
        selected.add(rule);
    }

    return (selected.isEmpty())?null:selected;
  }

  /**
   * Checks if two numeric intervals overlap.
   *
   * @param range  the first interval [lower, upper]
   * @param limits the second interval [lower, upper]
   * @return true if the intervals overlap, false otherwise
   */
  public static boolean doIntervalsOverlap(double range[], double limits[]) {
    if (range==null)
      return limits==null;
    if (limits==null)
      return false;
    // Handle infinite bounds
    boolean rangeLowerInfinite = Double.isInfinite(range[0]) && range[0] < 0;
    boolean rangeUpperInfinite = Double.isInfinite(range[1]) && range[1] > 0;
    boolean limitsLowerInfinite = Double.isInfinite(limits[0]) && limits[0] < 0;
    boolean limitsUpperInfinite = Double.isInfinite(limits[1]) && limits[1] > 0;
    // Check if the intervals do not overlap
    if (!rangeUpperInfinite && !limitsLowerInfinite && range[1] <= limits[0]) {
      return false;
    }
    if (!limitsUpperInfinite && !rangeLowerInfinite && limits[1] <= range[0]) {
      return false;
    }

    // If none of the above conditions are met, the intervals overlap
    return true;
  }
  /**
   * Checks if the limits interval completely includes the range interval.
   *
   * @param range  the first interval [lower, upper] to be checked for inclusion
   * @param limits the second interval [lower, upper] which is the boundary
   * @return true if limits includes range, false otherwise
   */
  public static boolean doesLimitsIncludeRange(double[] range, double[] limits) {
    if (range==null)
      return limits==null;
    if (limits==null)
      return false;

    // Handle infinite bounds
    boolean rangeLowerInfinite = Double.isInfinite(range[0]) && range[0] < 0;
    boolean rangeUpperInfinite = Double.isInfinite(range[1]) && range[1] > 0;
    boolean limitsLowerInfinite = Double.isInfinite(limits[0]) && limits[0] < 0;
    boolean limitsUpperInfinite = Double.isInfinite(limits[1]) && limits[1] > 0;

    // Check if the lower bound of range is within limits
    if (!limitsLowerInfinite)
      if (rangeLowerInfinite)
        return false;
      else
        if (range[0] < limits[0])
          return false;

    // Check if the upper bound of range is within limits
    if (!limitsUpperInfinite)
      if (rangeUpperInfinite)
        return false;
      else
        if (range[1] > limits[1])
          return false;

    // If none of the above conditions are met, range is included in limits
    return true;
  }

  /**
   * Settings from the last execution of the method getFeatureValuesDistributions()
   */
  private int nFeatureIntervals=10;
  private ArrayList<Double> resultBreaks=null;

  public ValuesFrequencies[][] getFeatureValuesDistributions(ArrayList<CommonExplanation> rules){
    return getFeatureValuesDistributions(rules,nFeatureIntervals,resultBreaks);
  }

  /**
   * For each feature, divides its full value range into intervals.
   * For each interval, counts the number of rules with this feature's range in the rule conditions
   * overlapping with this interval. This is done for each predicted class (in a case of classification rules)
   * or for intervals of predicted numeric values (in a case of regression model). For each class or
   * value intervals, the number of rules that do not involve this feature is also counted.
   *
   * @param rules - a (sub)set of rules for which to count frequencies. If null, takes the full set of rules
   *              from this instance of RuleSet.
   * @param nFeatureIntervals - in how many intervals the ranges of the feature values need to be divided.
   * @param resultBreaks - for a regression model, the breaks for dividing the range of the predicted values
   *                         into intervals. In a case of classification m odel, this argument is ignored.
   * @return 2D array of computed frequencies; 1st dimension: features; 2nd dimension: classes or result intervals
   */
  public ValuesFrequencies[][] getFeatureValuesDistributions(ArrayList<CommonExplanation> rules,
                                                             int nFeatureIntervals,
                                                             ArrayList<Double> resultBreaks) {
    if (rules==null)
      rules=this.rules;
    if (rules==null || attrMinMax==null)
      return null;
    if (!actionsDiffer && (Double.isNaN(minQValue) || minQValue>=maxQValue))
      return null;
    ArrayList<String> fNames=(orderedFeatureNames!=null && !orderedFeatureNames.isEmpty())?
                                 orderedFeatureNames:listOfFeatures;
    if (fNames==null) {
      fNames=new ArrayList<String>(attrMinMax.size());
      for (String name:attrMinMax.keySet())
        fNames.add(name);
    }
    else {
      fNames=(ArrayList<String>) fNames.clone();
      for (int j=fNames.size()-1; j>=0; j--)
        if (!attrMinMax.containsKey(fNames.get(j)))
          fNames.remove(j);
    }
    int nFeatures=fNames.size();

    if (!actionsDiffer && (resultBreaks==null || resultBreaks.isEmpty())) {
      int nResultIntervals=5;
      if (resultBreaks==null)
        resultBreaks=new ArrayList<Double>(nResultIntervals);
      double delta=(maxQValue-minQValue)/nResultIntervals;
      resultBreaks.add(minQValue+delta);
      for (int i=1; i<nResultIntervals-1; i++)
        resultBreaks.add(resultBreaks.get(i-1)+delta);
    }
    int nClasses=(actionsDiffer)?maxAction-minAction+1:resultBreaks.size()+1;

    if (nFeatureIntervals<2) nFeatureIntervals=2;

    ValuesFrequencies freq[][]=new ValuesFrequencies[nFeatures][nClasses];
    for (int i=0; i<nFeatures; i++) {
      String featureName=fNames.get(i);
      float minmax[]=attrMinMax.get(featureName);
      int nBreaks=nFeatureIntervals-1;
      double fBreaks[]=new double[nBreaks];
      double intervalWidth=((double)minmax[1]-(double)minmax[0])/nBreaks;
      fBreaks[0]=minmax[0]+intervalWidth/2;
      for (int j=1; j<fBreaks.length; j++)
        fBreaks[j]=fBreaks[j-1]+intervalWidth;

      freq[i]=new ValuesFrequencies[nClasses];
      for (int j=0; j<nClasses; j++) {
        freq[i][j]=new ValuesFrequencies();
        freq[i][j].featureName = featureName;
        freq[i][j].breaks=fBreaks;
        if (actionsDiffer)
          freq[i][j].action =minAction+j;
        else {
          freq[i][j].resultMinMax=new double[2];
          freq[i][j].resultMinMax[0]=(j==0)?minQValue:resultBreaks.get(j-1);
          freq[i][j].resultMinMax[1]=(j<resultBreaks.size())?resultBreaks.get(j):maxQValue;
          freq[i][j].lowHigh=(j==0)?-1:(j<resultBreaks.size())?0:1;
        }
      }
    }

    int classSizes[]=new int[nClasses];
    for (int i=0; i<nClasses; i++)
      classSizes[i]=0;

    for (CommonExplanation rule:rules) {
      int clIdx=(actionsDiffer)?rule.action-minAction:-1;
      if (clIdx<0) {
        if (Double.isNaN(rule.meanQ))
          continue;
        for (clIdx=0; clIdx<resultBreaks.size(); clIdx++)
          if (rule.meanQ<resultBreaks.get(clIdx))
            break;
      }
      ++classSizes[clIdx];
      for (int fIdx=0; fIdx<nFeatures; fIdx++)
        freq[fIdx][clIdx].countFeatureValuesInterval(rule.getFeatureInterval(fNames.get(fIdx)));
    }

    for (int j=0; j<nClasses; j++)
      for (int i=0; i<nFeatures; i++)
        freq[i][j].classSize =classSizes[j];

    this.nFeatureIntervals=nFeatureIntervals;
    this.resultBreaks=resultBreaks;

    return freq;
  }

  public void setNonSubsumed(boolean nonSubsumed) {
    this.nonSubsumed = nonSubsumed;
  }

  public void setAggregated(boolean aggregated) {
    this.aggregated = aggregated;
  }

  public double getAccThreshold() {
    return accThreshold;
  }

  public void setAccThreshold(double accThreshold) {
    this.accThreshold = accThreshold;
  }

  public double getMaxQDiff() {
    return maxQDiff;
  }

  public void setMaxQDiff(double maxQDiff) {
    this.maxQDiff = maxQDiff;
  }

  public boolean isExpanded() {
    return expanded;
  }

  public void setExpanded(boolean expanded) {
    this.expanded = expanded;
  }

  public void setTitle(String title) {
    this.title = title;
  }
  
  public boolean isFeaturesAreBinary() {
    return featuresAreBinary;
  }
  
  public void setFeaturesAreBinary(boolean featuresAreBinary) {
    this.featuresAreBinary = featuresAreBinary;
  }
  
  public void setOrderedFeatureNames(ArrayList<String> orderedFeatureNames) {
    if (orderedFeatureNames!=null && orderedFeatureNames.isEmpty())
      orderedFeatureNames=null;
    this.orderedFeatureNames = orderedFeatureNames;
  }

  public CommonExplanation getRule(int idx) {
    if (rules==null || idx>=rules.size() || idx<0)
      return null;
    return rules.get(idx);
  }

  public void addChild(RuleSet rs) {
    if (rs==null)
      return;
    rs.parent=this;
    if (children==null)
      children=new ArrayList<RuleSet>(20);
    children.add(rs);
    rs.versionLabel=versionLabel+"."+children.size();

    rs.minAction=minAction; rs.maxAction=maxAction; rs.actionsDiffer=actionsDiffer;
    rs.minQValue=minQValue; rs.maxQValue=maxQValue;
    rs.attrMinMax=attrMinMax;
    rs.featuresAreBinary=featuresAreBinary;
    rs.listOfFeatures=listOfFeatures;
    rs.orderedFeatureNames=orderedFeatureNames;
  }

  public RuleSet getOriginalRuleSet() {
    RuleSet rs=this;
    while (rs.parent!=null)
      rs=rs.parent;
    return rs;
  }

  public ArrayList<String> getSelectedFeatures(){
    if (featuresInDistances==null || featuresInDistances.isEmpty())
      return listOfFeatures;
    ArrayList<String> selected=new ArrayList<String>(featuresInDistances.size());
    for (int i=0; i<listOfFeatures.size(); i++)
      if (featuresInDistances.contains(listOfFeatures.get(i)))
        selected.add(listOfFeatures.get(i));
    return selected;
  }

  public static RuleSet createInstance(ArrayList<CommonExplanation> rules,
                                       boolean isNotSubsumed, boolean isAggregated, boolean isExpanded,
                                       double aggMinCoherence, double maxQDiff) {
    RuleSet rs=new RuleSet();
    rs.rules=rules;
    rs.nonSubsumed=isNotSubsumed;
    rs.aggregated=isAggregated;
    rs.expanded=isExpanded;
    rs.accThreshold=aggMinCoherence;
    rs.maxQDiff=maxQDiff;
    return rs;
  }

  public static RuleSet createInstance(ArrayList<CommonExplanation> rules,
                                       boolean isNotSubsumed, boolean isAggregated, boolean isExpanded) {
    return createInstance(rules,isNotSubsumed,isAggregated,isExpanded,
        1,0);
  }

  public static RuleSet createInstance(ArrayList rules) {
    return createInstance(rules,false,false,false,
        1,0);
  }
}
