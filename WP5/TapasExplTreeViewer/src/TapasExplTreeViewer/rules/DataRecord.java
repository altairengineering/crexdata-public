package TapasExplTreeViewer.rules;

import java.util.HashMap;

public class DataRecord {
  public static final byte NO_TARGET =0, CLASS_TARGET =1, VALUE_TARGET =2, RANGE_TARGET =3;

  public String id=null, name=null;
  public int idx=-1; //index in a data set or table
  public String trueClassLabel=null;
  public int origClassIdx =-1, predictedClassIdx=-1;
  public double origValue =Double.NaN, predictedValue=Double.NaN;
  public double origValueRange[]=null, predictedValueRange[]=null;
  public byte predictionType= NO_TARGET;
  public HashMap<String,DataElement> items=null;
  public HashMap<Integer,Integer> predictions=null;
  /**
   * To compare predictions made by different versions of a model,
   * a new version of each DataRecord is created before applying a model.
   * The field previousVersion keeps a reference to the previous version
   * of this data record.
   */
  public DataRecord previousVersion=null;

  public DataRecord(){}

  public DataRecord(int idx, String id, String name) {
    this.idx=idx; this.id=id; this.name=name;
  }
  /**
   * Creates a new instance of DataRecord with a reference to this instance.
   * Copies all fields, except for the original classes or values and predictions.
   * If predictions are available, they are stored as the original classes or values.
   * The fields for the predictions are left in the  "unknown" state.
   * The data items are not copied; the same hashmap with the items is passed to the new instance.
   * @return a new instance of DataRecord with a reference to this instance.
   */
  public DataRecord makeNewVersion() {
    DataRecord r=new DataRecord(idx,id,name);
    r.previousVersion=this;
    r.trueClassLabel=trueClassLabel;
    r.origClassIdx=(predictedClassIdx>=0)?predictedClassIdx:origClassIdx;
    r.origValue=(Double.isNaN(predictedValue))?origValue:predictedValue;
    r.origValueRange=(predictedValueRange==null)?origValueRange:predictedValueRange;
    r.predictionType=predictionType;
    r.items=items;
    return r;
  }

  public void addDataElement(DataElement item) {
    if (item==null || item.feature==null)
      return;
    if (items==null)
      items=new HashMap<String, DataElement>(50);
    items.put(item.feature,item);
  }

  public void addDataElement(String feature, String value, byte dataType) {
    if (feature==null)
      return;
    DataElement item=new DataElement();
    item.feature=feature;
    if (dataType>=0)
      item.setDataType(dataType);
    item.setStringValue(value);
    addDataElement(item);
  }

  public DataElement getDataElement(String feature) {
    if (items==null)
      return null;
    return items.get(feature);
  }

  public void erasePrediction(){
    predictedClassIdx=-1;
    predictedValue=Double.NaN;
    predictedValueRange=null;
    predictionType= NO_TARGET;
    if (predictions!=null)
      predictions.clear();
  }

  public byte getPredictionType() {
    if (predictionType> NO_TARGET)
      return predictionType;
    predictionType=(predictedClassIdx>=0)? CLASS_TARGET :
                    (!Double.isNaN(predictedValue))? VALUE_TARGET :
                    (predictedValueRange!=null)? RANGE_TARGET : NO_TARGET;
    return predictionType;
  }

  public byte getTargetType() {
    if (trueClassLabel!=null || origClassIdx >=0)
      return CLASS_TARGET;
    if (!Double.isNaN(origValue))
      return VALUE_TARGET;
    return NO_TARGET;
  }

  public boolean hasCorrectPrediction(){
    if (getPredictionType()==NO_TARGET)
      return false;
    if (predictionType==CLASS_TARGET)
      return predictedClassIdx== origClassIdx;
    if (predictionType==VALUE_TARGET)
      return !Double.isNaN(predictedValue) && !Double.isNaN(origValue) && predictedValue== origValue;
    if (predictionType==RANGE_TARGET)
      return predictedValueRange!=null && !Double.isNaN(origValue) &&
          origValue >=predictedValueRange[0] && origValue <=predictedValueRange[1];
    return false;
  }
}
