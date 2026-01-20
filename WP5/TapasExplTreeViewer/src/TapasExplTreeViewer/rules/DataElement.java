package TapasExplTreeViewer.rules;

public class DataElement {
  public static final byte CATEGORY=0, INTEGER=1, REAL=2;

  public String feature=null;
  public byte dataType=CATEGORY;

  public String stringValue=null;
  public double doubleValue=Double.NaN;
  public int intValue=Integer.MIN_VALUE;

  public String getFeature() {
    return feature;
  }

  public void setFeature(String feature) {
    this.feature = feature;
  }

  public byte getDataType() {
    return dataType;
  }

  public void setDataType(byte dataType) {
    this.dataType = dataType;
  }

  public String getStringValue() {
    return stringValue;
  }

  public void setStringValue(String stringValue) {
    this.stringValue = stringValue;
    if (stringValue!=null && dataType!=CATEGORY)
      if (dataType==INTEGER)
        try {
          intValue=Integer.parseInt(stringValue);
        } catch (Exception ex) {}
      else
      if (dataType==REAL)
        try {
          doubleValue=Double.parseDouble(stringValue);
        } catch (Exception ex) {}
  }

  public double getDoubleValue() {
    if (!Double.isNaN(doubleValue))
      return doubleValue;
    if (intValue>Integer.MIN_VALUE)
      return intValue;
    if (stringValue!=null)
      try {
        doubleValue=Double.parseDouble(stringValue);
      } catch (Exception ex) {}
    return doubleValue;
  }

  public void setDoubleValue(double doubleValue) {
    this.doubleValue = doubleValue;
    if (!Double.isNaN(doubleValue))
      dataType=REAL;
  }

  public int getIntValue() {
    if (intValue>Integer.MIN_VALUE)
      return intValue;
    if (!Double.isNaN(doubleValue))
      intValue=(int)Math.round(doubleValue);
    else
    if (stringValue!=null)
      try {
        intValue=Integer.parseInt(stringValue);
      } catch (Exception ex) {}
    return intValue;
  }

  public void setIntValue(int intValue) {
    this.intValue = intValue;
    if (intValue>Integer.MIN_VALUE)
      dataType=INTEGER;
  }

  public boolean hasAnyValue(){
    return stringValue!=null || Double.isNaN(doubleValue) || intValue>Integer.MIN_VALUE;
  }
}
