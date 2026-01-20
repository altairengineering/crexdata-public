package TapasExplTreeViewer.ui;

import javax.swing.*;
import javax.swing.table.DefaultTableCellRenderer;
import java.awt.*;
import java.text.NumberFormat;

public class NumericCellRenderer extends DefaultTableCellRenderer {

  public double numValue=Double.NaN;

  public double minValue=Double.NaN, maxValue=Double.NaN;
  private NumberFormat numberFormat;

  /**
   * Constructor to initialize the NumericCellRenderer with min and max values.
   * @param minValue The minimum value in the data range.
   * @param maxValue The maximum value in the data range.
   */
  public NumericCellRenderer(double minValue, double maxValue) {
    this.minValue = minValue;
    this.maxValue = maxValue;
    this.numberFormat = NumberFormat.getNumberInstance(); // To format numbers
    this.numberFormat.setMaximumFractionDigits(2); // Set the number of decimal places for display
    setHorizontalAlignment(SwingConstants.RIGHT); // Right-align numbers
  }

  @Override
  protected void paintComponent(Graphics g) {
    if (!Double.isNaN(numValue) && !Double.isNaN(minValue) && !Double.isNaN(maxValue) && maxValue>minValue) {
      // Draw the background bar

      // Draw a light gray bar proportional to the value in the background
      g.setColor(new Color(220, 220, 220));
      if (minValue>=0)
        g.fillRect(0,2,(int)Math.round(getWidth()*(numValue-minValue)/(maxValue-minValue)),getHeight()-4);
      else
      if (maxValue<=0) {
        int dw=(int) Math.round(getWidth()*(maxValue-numValue) / (maxValue - minValue));
        g.fillRect(getWidth()-dw, 2, dw, getHeight() - 4);
      }
      else { // min<=0, max>=0
        int xZero=(int)Math.round(getWidth()*(0-minValue)/(maxValue-minValue));
        if (numValue>0) {
          g.fillRect(xZero,2,(int)Math.round((getWidth()-xZero)*(numValue-0)/(maxValue-0)),getHeight()-4);
        }
        else {
          int dw=(int) Math.round(xZero*(0-numValue)/(0-minValue));
          g.fillRect(xZero-dw,2,dw,getHeight()-4);
        }
      }
    }
    // Call superclass to render the value text
    super.paintComponent(g);
  }

  @Override
  public void setValue(Object value) {
    numValue=Double.NaN;
    if (value==null) {
      setText("");
      return;
    }
    if (value instanceof Number) {
      numValue=((Number) value).doubleValue();
      setText(numberFormat.format(value));
    }
    else {
      try {
        numValue=Double.parseDouble(value.toString());
      } catch (Exception ex) {}
      setText(value.toString());
    }
  }
}
