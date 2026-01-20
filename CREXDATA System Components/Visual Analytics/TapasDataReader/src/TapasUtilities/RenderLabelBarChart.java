package TapasUtilities;

import javax.swing.*;
import javax.swing.table.TableCellRenderer;
import java.awt.*;

public class RenderLabelBarChart extends JLabel_BarChart implements TableCellRenderer {
  protected boolean bModeTimeOfDay=false;

  public RenderLabelBarChart(float min, float max) {
    super(min,max);
    setOpaque(false);
  }

  public void setbModeTimeOfDay() {
    bModeTimeOfDay=true;
  }

  public Component getTableCellRendererComponent(JTable table, Object value,
                                                 boolean isSelected, boolean hasFocus,
                                                 int row, int column) {
    Float v=Float.NaN;
    if (value!=null) {
      if (value instanceof Float)
        v=(Float)value;
      if (value instanceof Double)
        v=((Double)value).floatValue();
      if (value instanceof Integer)
        v=((Integer)value).floatValue();
    }
    if (value==null || v<min || v>max) {
      setText("");
      setValue(min-1);
    }
    else {
      setValue(v);
      if (bModeTimeOfDay)
        setText(String.format("%02f", v / 60) + ":" + String.format("%02f", v % 60));
      else
        if (value instanceof Integer)
          setText(""+((Integer)value));
        else
          setText(String.format("%.3f",v));
    }
    if (isSelected)
      setBackground(table.getSelectionBackground());
    else
      setBackground(table.getBackground());
    return this;
  }
}
