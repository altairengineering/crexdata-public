package TapasExplTreeViewer.vis;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.util.Map;

public class HeatmapDrawer extends JPanel {

  private String title=null, xAxisLabel=null, yAxisLabel=null;
  private int counts[][]=null;
  private int absMax=0;
  private int nPresent[]=null, nAbsent[]=null, maxSum=0;
  private double breaks[][]=null;
  private String[] yLabels=null; // Labels for y-axis (e.g., features or classes)
  private double[][] frequencies=null; // 2D array of frequencies

  private int leftMargin = 100, rightMargin=0, topMargin=0, bottomMargin=15; // Margins for labels and axes
  private int cellWidth=0, cellHeight=0; //to be computed when drawn

  public static Color minColor = new Color(255, 255, 200); // Light yellow
  public static Color maxColor = new Color(150, 0, 0); // Dark red
  public static Color colorFeatureMiss=new Color(200, 200, 200); // Color for rules not including the feature
  public static Color colorFeatureUsed=new Color(120, 180, 240); // Color for rules including the feature

  /**
   * Constructor for the heatmap drawer.
   *
   * @param xAxisLabel Label for the x-axis.
   * @param yAxisLabel Label for the y-axis.
   * @param counts 2D array of counts, to be transformed to frequencies from 0 to 1.
   * @param yLabels Labels for y-axis (e.g., features or classes).
   */
  public HeatmapDrawer(int[][] counts, int absMaxCount,
                       int nPresent[], int nAbsent[],
                       double breaks[][],
                       String title,
                       String xAxisLabel,
                       String yAxisLabel,
                       String[] yLabels) {
    this.counts=counts;
    this.nPresent=nPresent; this.nAbsent=nAbsent;
    this.breaks=breaks;
    this.title=title;
    this.xAxisLabel = xAxisLabel;
    this.yAxisLabel = yAxisLabel;
    this.yLabels = yLabels;
    
    updateData(counts,absMaxCount,nPresent,nAbsent);

    setToolTipText(""); // Enables the tooltip mechanism
    addMouseMotionListener(new MouseMotionAdapter() {
      @Override
      public void mouseMoved(MouseEvent e) {
        updateTooltip(e);
      }
    });
  }
  
  public void updateData(int[][] counts, int absMaxCount,
                         int nPresent[], int nAbsent[]) {
    if (counts==null)
      return;
    this.counts=counts; this.nPresent=nPresent; this.nAbsent=nAbsent;
    maxSum=0;
    if (absMaxCount<=0)
      for (int i=0; i<counts.length; i++)
        for (int j=0; j<counts[i].length; j++)
          if (counts[i][j]>absMaxCount)
            absMaxCount=counts[i][j];
    this.absMax=absMaxCount;
  
    this.frequencies = new double[counts.length][counts[0].length];
    for (int i=0; i<counts.length; i++)
      for (int j=0; j<counts[i].length; j++)
        frequencies[i][j]=(absMaxCount==0)?0:1.0*counts[i][j]/absMaxCount;
      
    if (isShowing())
      repaint();
  }

  public Dimension getMinumumSize() {
    return new Dimension(100+frequencies[0].length*5,15+frequencies.length*10);
  }

  @Override
  protected void paintComponent(Graphics g) {
    super.paintComponent(g);
    Graphics2D g2 = (Graphics2D) g;

    int panelWidth = getWidth();
    int panelHeight = getHeight();

    int heatmapWidth = panelWidth - leftMargin-rightMargin;
    int heatmapHeight = panelHeight - topMargin  - bottomMargin;

    cellWidth = heatmapWidth / frequencies[0].length;
    cellHeight = heatmapHeight / frequencies.length;

    if (cellWidth>2 && cellHeight>2) {
      // Draw heatmap cells
      for (int row = 0; row < frequencies.length; row++) {
        for (int col = 0; col < frequencies[0].length; col++) {
          double value = frequencies[row][col];
          Color cellColor = (value == 0) ? Color.white : interpolateColor(minColor, maxColor, value);
          g2.setColor(cellColor);
          int x = leftMargin + col * cellWidth;
          int y = topMargin + row * cellHeight;
          g2.fillRect(x, y, cellWidth, cellHeight);
        }
      }

      g2.setColor(new Color(192, 192, 192, 128));

      // Draw grid lines
      for (int row = 0; row <= frequencies.length; row++) {
        int y = topMargin + row * cellHeight;
        g2.drawLine(leftMargin, y, leftMargin + frequencies[0].length * cellWidth, y);
      }
      /*
      for (int col = 0; col <= frequencies[0].length; col++) {
        int x = leftMargin + col * cellWidth;
        g2.drawLine(x, topMargin, x, topMargin + heatmapHeight);
      }
      */
    }

    g2.setColor(Color.black);
    FontMetrics fm = g2.getFontMetrics();

    // Draw y-axis labels
    for (int row = 0; row < yLabels.length; row++) {
      int x = leftMargin - 2;
      int y = topMargin + row * cellHeight + cellHeight / 2 + fm.getAscent() / 2;
      if (nPresent!=null && nAbsent!=null) {
        if (maxSum<=0) {
          maxSum=0;
          for (int i=0; i<nPresent.length; i++){
            int sum=nPresent[i]+nAbsent[i];
            if (maxSum<sum)
              maxSum=sum;
          }
        }
        float sum=nPresent[row]+nAbsent[row];
        int maxBarW=leftMargin-4, barWidth=Math.round(sum/maxSum*maxBarW),
            includeWidth =Math.round(nPresent[row]/sum*barWidth);
        // Draw bar
        g2.setColor(colorFeatureMiss); // Color for rules not including the feature
        g2.fillRect(x-barWidth, topMargin + row * cellHeight + cellHeight / 4,
            barWidth, cellHeight / 2);
        g2.setColor(colorFeatureUsed); // Color for rules including the feature
        g2.fillRect(x-includeWidth, topMargin + row * cellHeight + cellHeight / 4,
            includeWidth, cellHeight / 2);
        g2.setColor(Color.black);
      }
      g2.drawString(yLabels[row], x - fm.stringWidth(yLabels[row]), y);
    }

    // Draw axis titles
    if (xAxisLabel!=null)
      g2.drawString(xAxisLabel, leftMargin + heatmapWidth / 2 - fm.stringWidth(xAxisLabel) / 2,
          panelHeight - bottomMargin+fm.getAscent());
    if (yAxisLabel!=null) {
      g2.rotate(-Math.PI / 2);
      g2.drawString(yAxisLabel, -panelHeight / 2 - fm.stringWidth(yAxisLabel) / 2, 20);
      g2.rotate(Math.PI / 2);
    }
  }

  /**
   * Interpolates between two colors based on a value between 0 and 1.
   *
   * @param minColor The color representing the minimum value.
   * @param maxColor The color representing the maximum value.
   * @param value A value between 0 and 1.
   * @return The interpolated color.
   */
  private Color interpolateColor(Color minColor, Color maxColor, double value) {
    value = Math.max(0, Math.min(1, value)); // Clamp value to [0, 1]
    int red = (int) (minColor.getRed() + value * (maxColor.getRed() - minColor.getRed()));
    int green = (int) (minColor.getGreen() + value * (maxColor.getGreen() - minColor.getGreen()));
    int blue = (int) (minColor.getBlue() + value * (maxColor.getBlue() - minColor.getBlue()));
    return new Color(red, green, blue);
  }

  private void updateTooltip(MouseEvent e) {
    setToolTipText(null);
    if (cellWidth<2 || cellHeight<2 || e.getY()<topMargin)
      return;

    int row = (e.getY() - topMargin) / cellHeight;
    if (row<0 || row>=counts.length)
      return;

    String rowLabel = yLabels[row];

    if (e.getX()<leftMargin) {
      if (nPresent!=null && nAbsent!=null)
        setToolTipText(formatTooltip(title,rowLabel,nPresent[row],nAbsent[row]));
    }
    else {
      int col = (e.getX() - leftMargin) / cellWidth;
      if (col >= 0 && col < counts[0].length) {
        double min = 0, max = 0;
        if (breaks != null) {
          min = (col == 0) ? Double.NEGATIVE_INFINITY : breaks[row][col - 1];
          max = (col >= breaks[row].length) ? Double.POSITIVE_INFINITY : breaks[row][col];
        }
        double value = frequencies[row][col];
        Color cellColor = (value == 0) ? Color.white : interpolateColor(minColor, maxColor, value);
        setToolTipText(formatTooltip(title, rowLabel, min, max, counts[row][col], nPresent[row], cellColor));
      }
    }
  }

  private String formatTooltip(String title, String rowLabel, int nInclude, int nMiss) {
    int sum=nInclude+nMiss;
    return String.format(
        "<html><body style=background-color:rgb(255,255,204)>" +
            "<h3>%s + %s:</h3>" +
            "<table><tr><td style=background-color:rgb(%d,%d,%d)>" +
            "Number of rules involving this feature:</td><td><b>%d</b></td><td><b>%.2f</b>%%</td></tr>" +
            "<tr><td style=background-color:rgb(%d,%d,%d)>" +
            "Number of rules without this feature:</td><td><b>%d</b></td><td><b>%.2f</b>%%</td></tr>" +
            "<tr><td>Total number of rules for this class:</td><td><b>%d</b></td></tr>" +
            "</table></body></html>",
        title, rowLabel,
        colorFeatureUsed.getRed(), colorFeatureUsed.getGreen(), colorFeatureUsed.getBlue(),
        nInclude,100.0*nInclude/sum,
        colorFeatureMiss.getRed(), colorFeatureMiss.getGreen(), colorFeatureMiss.getBlue(),
        nMiss,100.0*nMiss/sum,
        sum
    );
  }

  private String formatTooltip(String title, String rowLabel, double minVal,
                               double maxVal, int count, int nInclude, Color cellColor) {
    return String.format(
        "<html><body style=background-color:rgb(255,255,204)>" +
            "<h3>%s + %s:</h3>" +
            "<table><tr><td>Interval of feature values:</td><td>[<b>%.3f..%.3f</b>)</td></tr>" +
            "<tr><td>Count of rules: </td><td><b>%d</b> out of <b>%d</b> (max = %d)</td><td style=background-color:rgb(%d,%d,%d);" +
            "color:rgb(%d,%d,%d)>_____</td></tr>" +
            "</table></body></html>",
        title, rowLabel, minVal, maxVal, count, nInclude, absMax,
        cellColor.getRed(), cellColor.getGreen(), cellColor.getBlue(), cellColor.getRed(), cellColor.getGreen(), cellColor.getBlue()
    );
  }
}
