package TapasExplTreeViewer.rules;

/**
 * Contains frequencies for intervals of feature values.
 */
public class ValuesFrequencies {
  public String featureName=null;
  /**
   * If the frequencies correspond to a particular predicted class or action in a classification model
   */
  public int action =-1;
  /**
   * If the frequencies correspond to an interval of predicted values in a regression model
   */
  public double resultMinMax[]=null;
  /**
   * Total count of the rules predicting the given action or values from the given interval
   */
  public int classSize =0;
  /**
   * Whether this interval of resulting values is the lowest (-1), the highest (1), or in between
   */
  public int lowHigh=0;
  /**
   * Breaks between the intervals of the feature values. The number of the breaks equals
   * the number of intervals - 1.
   */
  public double breaks[]=null;
  /**
   * Frequencies (counts of rules) by the intervals
   */
  public int counts[]=null;
  /**
   * Count of rules that do not involve this feature
   */
  public int nAbsences=0;
  /**
   * Number of rules that have been counted
   */
  public int nRulesWithFeature =0;

  public void countFeatureValuesInterval(double fromTo[]) {
    if (fromTo==null)
      ++nAbsences;
    else {
      ++nRulesWithFeature;
      if (breaks==null)
        return;
      if (counts==null) {
        counts=new int[breaks.length+1];
        for (int i=0; i<counts.length; i++)
          counts[i]=0;
      }
      int i0=0;
      if (!Double.isInfinite(fromTo[0]))
        for (int i=0; i<breaks.length && fromTo[0]>=breaks[i]; i++)
          i0=i+1;
      int iEnd=(Double.isInfinite(fromTo[1]))?breaks.length:i0;
      for (int i=iEnd; i<breaks.length && fromTo[1]>breaks[i]; i++)
        iEnd=i+1;
      for (int i=i0; i<=iEnd; i++)
        ++counts[i];
    }
  }
}
