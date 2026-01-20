package TapasExplTreeViewer.vis;

import TapasExplTreeViewer.util.CoordinatesReader;
import TapasExplTreeViewer.util.MatrixWriter;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.io.File;
import java.util.ArrayList;

public class TSNE_Runner implements ProjectionProvider{
  public static int nInstancesCreated=0;
  
  public int currInstanceN=-1;
  /**
   * Matrix of distances between the objects to project and show
   */
  public double distances[][]=null;
  /**
   * The value of the "perplexity" parameter
   */
  public int perplexity=30;
  /**
   * The projection obtained (updated iteratively)
   */
  protected double proj[][]=null;
  /**
   * When a file is created, it is registered in this list, to be deleted afterwards
   */
  public ArrayList<File> createdFiles=null;
  protected boolean distanceMatrixPutInFile=false;
  
  protected ArrayList<ChangeListener> changeListeners=null;
  
  public TSNE_Runner() {
    currInstanceN=++nInstancesCreated;
  }
  
  public void addChangeListener(ChangeListener l) {
    if (changeListeners==null)
      changeListeners=new ArrayList(5);
    if (!changeListeners.contains(l))
      changeListeners.add(l);
  }
  
  public void removeChangeListener(ChangeListener l) {
    if (l!=null && changeListeners!=null)
      changeListeners.remove(l);
  }
  
  public void notifyChange(){
    if (changeListeners==null || changeListeners.isEmpty())
      return;
    ChangeEvent e=new ChangeEvent(this);
    for (ChangeListener l:changeListeners)
      l.stateChanged(e);
  }
  
  public int getPerplexity() {
    return perplexity;
  }
  
  public void setPerplexity(int perplexity) {
    this.perplexity = perplexity;
  }
  
  public void setDistanceMatrix(double distances[][]) {
    this.distances = distances;
    runAlgorithm();
  }
  
  public void setFileRegister(ArrayList<File> createdFiles) {
    this.createdFiles=createdFiles;
  }
  
  public void runAlgorithm() {
    if (distances != null) {
      System.out.println("Running the t-SNE algorithm");
      String numSt=String.format("%02d",currInstanceN), distFName="distances_"+numSt;
      SwingWorker worker=new SwingWorker() {
        @Override
        public Boolean doInBackground(){
          String pathName=distFName+".csv";
          File matrFile=new File(pathName);
          if (!distanceMatrixPutInFile || !matrFile.exists()) {
            MatrixWriter.writeMatrixToFile(distances, pathName, true);
            if (createdFiles != null)
              createdFiles.add(matrFile);
            distanceMatrixPutInFile=true;
          }
          //String command="cmd.exe /C TSNE-precomputed.bat "+distFName+" "+perplexity;
          String command = "cmd.exe",
              argument = "/C",
              batchFilePath = "TSNE-precomputed.bat "+distFName+" "+perplexity; // ioFilePath + "TSNE-" + numSt + ".bat";
          try {
            ProcessBuilder pb = new ProcessBuilder(command, argument, batchFilePath);
            pb.redirectOutput(ProcessBuilder.Redirect.INHERIT); // Redirect output to Java's standard output
            pb.redirectError(ProcessBuilder.Redirect.INHERIT);  // Redirect error to Java's standard error
            Process p = pb.start(); //Runtime.getRuntime().exec(command);
            int exit_value = p.waitFor();
            System.out.println("TSNE: finished, code="+exit_value);
            pathName=distFName+"_out_p"+perplexity+".csv";
            proj= CoordinatesReader.readCoordinatesFromFile(pathName);
            if (createdFiles!=null)
              createdFiles.add(new File(pathName));
            notifyChange();
          } catch (Exception e) {
            e.printStackTrace();
          }
          return true;
        }
        @Override
        protected void done() {
          //notifyChange();
        }
      };
      worker.execute();
    }
    
  }
  
  public double[][] getProjection(){
    return proj;
  }
  
  public String getProjectionTitle() {
    return "t-SNE projection; perplexity = "+perplexity;
  }
}
