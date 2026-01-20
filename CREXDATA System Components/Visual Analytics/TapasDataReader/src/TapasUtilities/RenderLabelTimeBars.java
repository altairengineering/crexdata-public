package TapasUtilities;

import javax.swing.*;
import javax.swing.table.TableCellRenderer;
import java.awt.*;

public class RenderLabelTimeBars extends JLabel_TimeBars implements TableCellRenderer {
  public RenderLabelTimeBars (float max) {
    super(max);
    setOpaque(false);
  }
  public Component getTableCellRendererComponent (JTable table, Object value, boolean isSelected, boolean hasFocus, int row, int column) {
    setValue((int[])value);
    if (isSelected) {
      setForeground(Color.black);
      setBackground(table.getSelectionBackground());
    }
    else {
      setForeground(Color.darkGray);
      setBackground(table.getBackground());
    }
    return this;
  }
}
