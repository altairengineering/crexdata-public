package TapasExplTreeViewer.ui;

import javax.swing.*;
import javax.swing.table.TableCellRenderer;
import java.awt.*;
import java.util.Arrays;

public class JLabel_Subinterval extends JLabel implements TableCellRenderer {
  public double min=Double.NaN, max=Double.NaN, absMin=Double.NaN, absMax=Double.NaN;
  boolean isInteger=false;
  public double v[]=null, values[]=null, Q1,Q2,Q3,avg;
  public boolean drawTexts=false, drawValues=false, drawStats=false;
  public int precision=0;
  public Color barColor=new Color(160,160,160);
  
  public JLabel_Subinterval() {
    setHorizontalAlignment(SwingConstants.RIGHT);
    setOpaque(false);
  }
  public void setDrawTexts (boolean drawTexts) {
    this.drawTexts=drawTexts;
  }
  public void setDrawValues (boolean drawValues) {
    this.drawValues=drawValues;
  }
  public void setDrawStats (boolean drawStats) {
    this.drawStats=drawStats;
  }

  public void setPrecision(int precision) {
    this.precision = precision;
  }

  public void setValues (double v[]) {
    if (v!=null && v.length>=4) {
      min=v[0]; max=v[1];
      absMin=v[2]; absMax=v[3];
      isInteger= Math.floor(absMin)==Math.ceil(absMin) && Math.floor(absMax)==Math.ceil(absMax);
    }
    this.v=v;
    if (v.length>5 && drawStats) {
      values=new double[v.length-4];
      for (int i=4; i<v.length; i++)
        values[i-4]=v[i];
      Arrays.sort(values);
      avg=0;
      for (int i=0; i<values.length; i++)
        avg+=values[i];
      avg/=values.length;
      Q1=quartile(values,25);
      Q2=quartile(values,50);
      Q3=quartile(values,75);
    }
    else
      values=null;
    if (drawTexts) {
      if (Double.isNaN(min) || Double.isNaN(max))
        setText("");
      else
        //setText(Math.round(min)+".."+Math.round(max));
        setText(String.format("%."+precision+"f..%."+precision+"f",min,max));
    }
    else
      setText("");
  }
  public void paint (Graphics g) {
    if (Double.isNaN(min) || Double.isNaN(max) || Double.isNaN(absMin) || Double.isNaN(absMax)) {
      super.paint(g);
      return;
    }
    int w=getWidth(), h=getHeight();
    g.setColor(getBackground());
    g.fillRect(0, 0, w, h);
    int x1 = (int) Math.round((min - absMin) * w / (absMax - absMin)),
        x2 = (int) Math.round((max - absMin) * w / (absMax - absMin));
    int barW=Math.max(1,x2 - x1);
    if (isInteger) {
      int step = (int) Math.round(w / (absMax - absMin + 1));
      if (step>1) {
        x1 = step * (int) Math.round(min - absMin);
        barW = step * (int) Math.round(max - min + 1);
        x2 = x1 + barW;
      }
    }
    g.setColor(barColor);
    g.fillRect(x1, h / 2, barW, h / 2);
    g.setColor(Color.gray.darker());
    if (drawValues)
      for (int i=4; i<v.length; i++) {
        int x=(int) Math.round((v[i] - absMin) * (w-1) / (absMax - absMin));
        g.drawLine(x,h/2, x, 3*h/4);
      }
    if (drawStats && values!=null) {
      int dy=2;
      x1=(int) Math.round((values[0] - absMin) * (w-1) / (absMax - absMin));
      x2=(int) Math.round((values[values.length-1] - absMin) * (w-1) / (absMax - absMin));
      g.drawLine(x1,h/2-dy, x2, h/2-dy);
      x1=(int) Math.round((Q1 - absMin) * (w-1) / (absMax - absMin));
      x2=(int) Math.round((Q3 - absMin) * (w-1) / (absMax - absMin));
      g.drawLine(x1,h/2-dy-1, x2, h/2-dy-1);
      int x=(int) Math.round((Q2 - absMin) * (w-1) / (absMax - absMin));
      g.drawLine(x,h/2-dy-3, x, h/2-dy);
      x=(int) Math.round((avg - absMin) * (w-1) / (absMax - absMin));
      g.setColor(Color.blue);
      g.drawLine(x,0, x, h/2-dy);
    }
    setForeground(new Color(0,0,0,80));
    super.paint(g);
  }
  public Component getTableCellRendererComponent(JTable table,
                                                 Object value,
                                                 boolean isSelected,
                                                 boolean hasFocus,
                                                 int row, int column) {
    if (isSelected)
      setBackground(table.getSelectionBackground());
    else
      setBackground(table.getBackground());
    setValues((double[])value);
    return this;
  }

  /**
   * Retrive the quartile value from an array
   * .
   * @param sortedValues THe array of data
   * @param lowerPercent The percent cut off. For the lower quartile use 25,
   *      for the upper-quartile use 75
   * @return
   */
  public static double quartile(double[] sortedValues, double lowerPercent) {
    int n = Math.min(sortedValues.length-1, (int) Math.round(sortedValues.length * lowerPercent / 100));
    return sortedValues[n];
  }
}
