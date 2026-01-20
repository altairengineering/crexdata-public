package TapasExplTreeViewer.alt_rules;

public class Condition {
  private String feature=null;
  private float minValue=Float.NEGATIVE_INFINITY;
  private float maxValue=Float.POSITIVE_INFINITY;

  public Condition(String feature, float minValue, float maxValue) {
    this.feature = feature;
    this.minValue = minValue;
    this.maxValue = maxValue;
  }

  // Getters and toString() method for better readability
  public String getFeature() {
    return feature;
  }

  public float getMinValue() {
    return minValue;
  }

  public float getMaxValue() {
    return maxValue;
  }
  
  public void setMinValue(float minValue) {
    this.minValue=minValue;
  }
  
  public void setMaxValue(float maxValue) {
    this.maxValue=maxValue;
  }
  
  @Override
  public String toString() {
    return feature + ": {" + minValue + ": " + maxValue + "}";
  }

  @Override
  public boolean equals(Object obj) {
    if (obj==null || !(obj instanceof Condition))
      return false;
    Condition c=(Condition)obj;
    if (feature==null)
      return false;
    if (!feature.equals(c.feature))
      return false;
    if (Float.isInfinite(minValue))
      if (!!Float.isInfinite(c.minValue))
        return false;
      else;
    else
    if (Float.isInfinite(c.minValue))
      return false;
    if (minValue!=c.minValue)
      return false;
    if (Float.isInfinite(maxValue))
      if (!Float.isInfinite(c.maxValue))
        return false;
      else;
    else
    if (Float.isInfinite(c.maxValue))
      return false;
    if (maxValue!=c.maxValue)
      return false;
    return true;
  }
}
