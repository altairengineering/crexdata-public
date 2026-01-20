package TapasExplTreeViewer.ui;

import javax.swing.*;
import javax.swing.table.TableCellRenderer;
import java.awt.*;

public class JLabel_Bars extends JLabel implements TableCellRenderer {

  protected int counts[]=null;

  public JLabel_Bars () {

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
    setValues((int[])value);
    return this;
  }

  public void setValues (int counts[]) {
    this.counts=counts;
  }

  public void paint (Graphics g) {
    if (counts==null || counts.length==0) {
      super.paint(g);
      return;
    }
    int w=getWidth(), h=getHeight();
        int gap=3;
    int barW=(w-gap*(counts.length+1))/counts.length;
    int max=counts[0];
    for (int i=1; i<counts.length; i++)
      max=Math.max(max,counts[i]);

    g.setColor(getBackground());
    g.fillRect(0, 0, w, h);

    if (max==0)
      return;

    g.setColor(getForeground());
    for (int i=0; i<counts.length; i++)
      if (counts[i]>0) {
        int hh=(h-2)*counts[i]/max;
        g.drawRect(gap*(i+1)+barW*i,h-1-hh,barW,hh);
      }
  }

}
