package TapasDataReader;

public class CountMatrix {
  public String colNames[]=null;
  public String rowNames[]=null;
  public Integer cellValues[][]=null;
  
  public int getColumnMax(int col) {
    if (cellValues==null || cellValues[0]==null || col<1)
      return 0;
    if (rowNames!=null)
      --col; //the first (0th) column contains row names
    if (col<0 || col>=cellValues[0].length)
      return 0;
    int max=0;
    for (int i=0; i<cellValues.length; i++)
      if (max<cellValues[i][col])
        max=cellValues[i][col];
    return max;
  }
}
