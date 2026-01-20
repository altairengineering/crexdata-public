package TapasExplTreeViewer.util;

import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;

public class CoordinatesReader {
  public static File lastDir=null;
  
  public static String lastFileName=null;
  
  public static double[][] readCoordinatesFromChosenFile() {
    JFileChooser fileChooser = new JFileChooser();
    if (lastDir!=null)
      fileChooser.setCurrentDirectory(lastDir);
    FileNameExtensionFilter filter = new FileNameExtensionFilter("text files","txt","csv");
    fileChooser.setFileFilter(filter);
    int result = fileChooser.showOpenDialog(null);
    if (result == JFileChooser.APPROVE_OPTION) {
      lastDir=fileChooser.getCurrentDirectory();
      return readCoordinatesFromFile(fileChooser.getSelectedFile());
    }
    return null;
  }
  
  public static double[][] readCoordinatesFromFile(String pathName) {
    System.out.println("Trying to read coordinates from file "+pathName);
    return readCoordinatesFromFile(new File(pathName));
  }
  
  public static double[][] readCoordinatesFromFile(File file) {
    if (file==null || !file.exists()) {
      System.out.println("The file does not exist!");
      return null;
    }
    
    BufferedReader br = null;
    try {
      br = new BufferedReader(new FileReader(file));
    } catch (Exception ex) {
      System.out.println(ex);
      return null;
    }
  
    ArrayList<double[]> coordList=null;
    String st;
    try {
      while ((st = br.readLine()) != null) {
        String[] tokens=st.split(",");
        if (tokens==null || tokens.length<2)
          continue;
        if (coordList==null)
          coordList=new ArrayList<double[]>(3000);
        int idx=coordList.size();
        if (tokens.length>2 && tokens[0].length()>0) {
          //the string begins with the ordinal number (index) of the point
          try {
            idx=Integer.parseInt(tokens[0]);
          } catch (Exception ex) {}
        }
        double xy[]={Double.NaN,Double.NaN};
        for (int i=0; i<2; i++) {
          int j=(tokens.length==2)?i:i+1;
          try {
            xy[i]=Double.parseDouble(tokens[j]);
          } catch (Exception ex) {}
        }
        if (!Double.isNaN(xy[0]) && !Double.isNaN(xy[1]))
          coordList.add(idx,xy);
      }
      br.close();
    } catch (Exception ex) {
      System.out.println(ex.toString());
    }
    if (coordList==null || coordList.isEmpty())
      return null;
    double coords[][]=new double[coordList.size()][];
    for (int i=0; i<coordList.size(); i++)
      coords[i]=coordList.get(i);
    
    lastFileName=file.getName();
    return coords;
  }
}
