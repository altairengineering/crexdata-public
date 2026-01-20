package TapasExplTreeViewer.ui;

import TapasDataReader.CommonExplanation;

import javax.swing.*;
import javax.swing.table.TableCellRenderer;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.Vector;

public class JLabel_Rule extends JLabel implements TableCellRenderer {

  //BufferedImage image=null;
  CommonExplanation ex=null;
  java.util.Vector<String> attrs=null;
  Vector<float[]> minmax=null;

  public void setAttrs (Vector<String> attrs, Vector<float[]> minmax) {
    this.attrs=attrs;
    this.minmax=minmax;
  }

  public void paint (Graphics g) {
    //if (image==null || image.getWidth()!=getWidth() || image.getHeight()!=getHeight())
    BufferedImage image=ShowSingleRule.getImageForRule(getWidth(), getHeight(), ex, null, attrs, minmax);
    g.drawImage(image,0,0,null);
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
    ex=(CommonExplanation)value;
    return this;
  }

}
