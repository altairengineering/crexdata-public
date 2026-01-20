package TapasUtilities;

import javax.swing.*;
import java.awt.*;

public class JLabel_BarChart extends JLabel {
  protected float min,max,v;
  protected String text;

  public JLabel_BarChart(float min, float max) {
    super("", Label.RIGHT);
    this.min=min;
    this.max=max;
    setHorizontalAlignment(SwingConstants.RIGHT);
  }

  public void setMinMax(float min, float max) {
    this.min=min;
    this.max=max;
  }

  public void setValue (float v) {
    this.v=v;
  }

  public void setText (String text) {
    super.setText(text);
    this.text=text;
  }

  public void paint (Graphics g) {
    //System.out.println("* v="+v);
    g.setColor(getBackground());
    g.fillRect(0,0,getWidth(),getHeight());
    g.setColor(Color.lightGray);
    if (min>=0)
      g.fillRect(0,2,(int)Math.round(getWidth()*(v-min)/(max-min)),getHeight()-4);
    else
      if (max<=0) {
        int dw=(int) Math.round(getWidth()*(max-v) / (max - min));
        g.fillRect(getWidth()-dw, 2, dw, getHeight() - 4);
      }
      else { // min<=0, max>=0
        int xZero=(int)Math.round(getWidth()*(0-min)/(max-min));
        if (v>0) {
          g.fillRect(xZero,2,(int)Math.round((getWidth()-xZero)*(v-0)/(max-0)),getHeight()-4);
        }
        else {
          int dw=(int) Math.round(xZero*(0-v)/(0-min));
          g.fillRect(xZero-dw,2,dw,getHeight()-4);
        }
      }
    super.paint(g);
  }
}