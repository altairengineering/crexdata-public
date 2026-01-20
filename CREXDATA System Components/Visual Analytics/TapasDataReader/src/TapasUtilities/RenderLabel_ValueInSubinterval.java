package TapasUtilities;

import javax.swing.*;
import javax.swing.table.TableCellRenderer;
import java.awt.*;

public class RenderLabel_ValueInSubinterval extends JLabel_ValueInSubinterval implements TableCellRenderer {
  public RenderLabel_ValueInSubinterval() {
    super();
    setOpaque(false);
  }
  public Component getTableCellRendererComponent(JTable table, Object value, boolean isSelected, boolean hasFocus, int row, int column) {
    if (isSelected)
      setBackground(table.getSelectionBackground());
    else
      setBackground(table.getBackground());
    setValues((float[])value);
    return this;
  }

}
