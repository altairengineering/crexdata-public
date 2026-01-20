package TapasUtilities;

import javax.swing.*;
import java.awt.*;

public class JLabel_TimeLine extends JLabel {
  float max;
  int v[]=null;
  public JLabel_TimeLine (float max) {
    super("");
    this.max=max;
  }
  public void setValue (int v[]) {
    this.v=v;
  }
  public void paint (Graphics g) {
    g.setColor(getBackground());
    g.fillRect(0,0,getWidth(),getHeight());
    if (v!=null && v.length>0) {
      int x[] = new int[2 + v.length], y[] = new int[2 + v.length];
      x[0]=0; x[x.length-1]=getWidth();
      y[0]=y[y.length-1]=getHeight();
      for (int i=0; i<v.length; i++) {
        x[1+i]=(int)Math.round(getWidth()*(1f*i/(v.length-1)));
        y[1+i]=(int)Math.round(getHeight()*(1-1f*v[i]/max));
      }
      g.setColor(getForeground());
      g.fillPolygon(x, y, x.length);
    }
    super.paint(g);
  }
}
