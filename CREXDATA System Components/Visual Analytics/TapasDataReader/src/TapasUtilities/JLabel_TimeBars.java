package TapasUtilities;

import javax.swing.*;
import java.awt.*;

public class JLabel_TimeBars extends JLabel {
  float max;
  int v[]=null;
  public JLabel_TimeBars (float max) {
    super("");
    this.max=max;
  }
  public void setValue (int v[]) {
    this.v=v.clone();
    for (int i=this.v.length-1; i>0; i--)
      this.v[i]=this.v[i]-this.v[i-1];
  }
  public void paint (Graphics g) {
    g.setColor(getBackground());
    g.fillRect(0,0,getWidth(),getHeight());
    if (v!=null && v.length>0) {
      g.setColor(getForeground());
      for (int i = 0; i < v.length; i++)
        if (v[i] > 0) {
          int x = (int) Math.round(1f * getWidth() * i / v.length);
          int h = (int) Math.round(getHeight() * v[i] / max);
          g.drawLine(x, getHeight() - h, x, getHeight());
        }
    }
    super.paint(g);
  }
}
