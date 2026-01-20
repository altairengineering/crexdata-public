package TapasExplTreeViewer.vis;

import TapasExplTreeViewer.rules.ClassConfusionMatrix;

import javax.swing.*;
import java.awt.*;

public class ClassConfusionMatrixFrame extends JFrame {
  public JTabbedPane versionsPane=null;

  public ClassConfusionMatrixFrame(ClassConfusionMatrix cMatrix, String rulesInfoText) {
    JPanel p=makeMatrixPanel(cMatrix, rulesInfoText);
    if (p==null)
      return;
    setTitle("Class confusion matrix");
    setSize(800, 700);
    setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
    setLocationRelativeTo(null);

    versionsPane=new JTabbedPane();
    versionsPane.setTabPlacement(JTabbedPane.BOTTOM);
    versionsPane.addTab(cMatrix.data.versionLabel,p);
    setLayout(new BorderLayout());
    add(versionsPane,BorderLayout.CENTER);
  }

  public JPanel makeMatrixPanel(ClassConfusionMatrix cMatrix, String rulesInfoText) {
    if (cMatrix==null || cMatrix.classNs==null || cMatrix.counts==null || cMatrix.nDataTotal<1)
      return null;

    String labels[]=new String[cMatrix.classNs.size()];
    for (int i=0; i<cMatrix.classNs.size(); i++)
      labels[i]=cMatrix.classNs.get(i).toString();

    // Create the tabbed pane
    JTabbedPane tabbedPane = new JTabbedPane();

    // Create the MatrixPainter instance for the count matrix
    MatrixPainter countMatrixPainter = new MatrixPainter(cMatrix.counts);
    countMatrixPainter.setShowValues(true);
    countMatrixPainter.setShowColumnTotals(true);
    countMatrixPainter.setMinValue(0);
    countMatrixPainter.setLabels(labels);
    JScrollPane countScrollPane = new JScrollPane(countMatrixPainter);
    tabbedPane.addTab("Counts", countScrollPane);

    // Create the MatrixPainter instance for the percentage matrix
    MatrixPainter percentageMatrixPainter = new MatrixPainter(cMatrix.percents,false);
    percentageMatrixPainter.setShowValues(true);
    percentageMatrixPainter.setMinValue(0);
    percentageMatrixPainter.setLabels(labels);
    JScrollPane percentageScrollPane = new JScrollPane(percentageMatrixPainter);
    tabbedPane.addTab("Percentages", percentageScrollPane);

    JPanel p=new JPanel();
    p.setLayout(new BorderLayout());
    p.add(tabbedPane, BorderLayout.CENTER);

    JTextArea infoArea=new JTextArea("Rule set: "+rulesInfoText);
    infoArea.setLineWrap(true);
    infoArea.setWrapStyleWord(true);
    infoArea.append("\nResults of applying the rules to "+cMatrix.nDataTotal+" data records. ");
    if (cMatrix.data!=null && cMatrix.data.description!=null) {
      infoArea.append("\nDataVersion: "+cMatrix.data.description);
      if (cMatrix.data.previousVersion!=null && cMatrix.data.previousVersion.description!=null)
        infoArea.append("\nPrevious version of the data: "+cMatrix.data.previousVersion.description+"\n");
    }
    double accuracy=100.0*cMatrix.nSame/cMatrix.nDataTotal;
    infoArea.append("Number of classes: "+cMatrix.counts.length+
        "; number of correct class assignments: "+cMatrix.nSame+
        String.format(" (%.2f %%)",accuracy)+
        "; number of wrong class assignments: "+(cMatrix.nDataTotal-cMatrix.nSame)+
        String.format(" (%.2f %%)",100.0-accuracy));
    p.add(infoArea,BorderLayout.SOUTH);
    return p;
  }

  public void addMatrix(ClassConfusionMatrix cMatrix, String rulesInfoText) {
    JPanel p=makeMatrixPanel(cMatrix,rulesInfoText);
    if (p==null)
      return;
    versionsPane.addTab("version "+(1+versionsPane.getComponentCount()),p);
    versionsPane.setSelectedIndex(versionsPane.getComponentCount()-1);
  }
}
