package TapasExplTreeViewer.util;

import java.io.File;
import java.io.FileWriter;

public class MatrixWriter {
  public static boolean writeMatrixToFile(double data[][],
                                          String pathName,
                                          boolean scaleTo0_1) {
    if (data==null)
      return false;
    if (scaleTo0_1) {
      double max=Double.NaN;
      boolean hasNaN=false;
      for (int i=0; i<data.length; i++)
        for (int j=0; j<data[i].length; j++)
          if (!Double.isNaN(data[i][j]) && !Double.isInfinite(data[i][j]))
            if (Double.isNaN(max) || max<Math.abs(data[i][j]))
              max=Math.abs(data[i][j]);
            else;
          else
            hasNaN=true;
      if (hasNaN && !Double.isNaN(max) && max>0)
        max*=2;
      if (!Double.isNaN(max) && max>0 && max!=1)
        for (int i=0; i<data.length; i++)
          for (int j=0; j<data[i].length; j++)
            if (!Double.isNaN(data[i][j]) && !Double.isInfinite(data[i][j]))
              data[i][j]/=max;
            else
              data[i][j]=1;
    }
    File file=new File(pathName);
    try {
      file.createNewFile();
      FileWriter writer=new FileWriter(file);
      for (int i=0; i<data.length; i++) {
        StringBuffer sb=new StringBuffer();
        for (int j=0; j<data.length; j++) {
          if (j>0)
            sb.append(",");
          sb.append(data[i][j]);
        }
        writer.write(sb.toString()+"\r\n");
      }
      writer.close();
    } catch (Exception ex) {
      System.out.println(ex);
      return false;
    }
    return true;
  }
}
