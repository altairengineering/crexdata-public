package TapasExplTreeViewer.ui;

import TapasDataReader.CommonExplanation;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.Vector;

public class ShowSingleRule {

  public static BufferedImage getImageForRule (int w, int h, CommonExplanation ex,
                                               Vector<CommonExplanation> vex,
                                               Vector<String> attrs,
                                               Vector<float[]> minmax) {
    BufferedImage image=new BufferedImage(w,h,BufferedImage.TYPE_INT_ARGB);
    int offsetX=3, offsetY=2;
    int dx=(w-2*offsetX) / attrs.size(),
        dy=h-2*offsetY;
    offsetX=(w-attrs.size()*dx)/2;
    boolean useThickerLines=(w/ex.eItems.length>=14) && h>=40;
    int barW=(useThickerLines)?6:4;
    Graphics2D g = image.createGraphics();
    Stroke s=g.getStroke();
    for (int i=0; i<attrs.size(); i++) {
      g.setColor(Color.lightGray);
      int x=offsetX+i*dx+dx/2;
      g.drawLine(x,offsetY,x,offsetY+dy);
      boolean found=false;
      float absMin=minmax.get(i)[0], absMax=minmax.get(i)[1];
      boolean isInteger= Math.floor(absMin)==Math.ceil(absMin) && Math.floor(absMax)==Math.ceil(absMax);
      for (int j=0; j<ex.eItems.length && !found; j++)
        if (attrs.elementAt(i).equals(ex.eItems[j].attr)) {
          g.setColor(Color.darkGray);
          double vv[]=new double[2];
          int y[]=new int[2];
          for (int k=0; k<y.length; k++) {
            vv[k]=ex.eItems[j].interval[k];
            if (vv[k]==Double.NEGATIVE_INFINITY)
              vv[k]=absMin;
            if (vv[k]==Double.POSITIVE_INFINITY)
              vv[k]=absMax;
            y[k]=(int)Math.round(dy*(vv[k]-absMin)/(absMax-absMin));
          }
          if (useThickerLines)
            g.setStroke(new BasicStroke(2));
          int hh=Math.max(1,y[1]-y[0]-1);
          if (isInteger) {
            int step = (int) Math.round(dy / (absMax - absMin + 1));
            if (step>1) {
              y[0] = step * (int) Math.round(vv[0] - absMin);
              hh = step * (int) Math.round(vv[1] - vv[0] + 1);
              y[1] = y[0] + hh;
            }
          }
          g.drawRect(x-2,offsetY+dy-y[1],barW,hh);
          if (useThickerLines)
            g.setStroke(s);
          found=true;
        }
      if (vex!=null && !vex.isEmpty()) {
        Color cblue=new Color(0,127,127, 64+96/vex.size());
        for (int idx = 0; idx < vex.size(); idx++) {
          found = false;
          for (int j = 0; j < vex.elementAt(idx).eItems.length && !found; j++)
            if (attrs.elementAt(i).equals(vex.elementAt(idx).eItems[j].attr)) {
              g.setColor(cblue);
              double vv[]=new double[2];
              int y[] = new int[2];
              double min=vex.elementAt(idx).eItems[j].interval[0], max=vex.elementAt(idx).eItems[j].interval[1];
              for (int k = 0; k < y.length; k++) {
                vv[k] = (k==0)?min:max;
                if (vv[k] == Double.NEGATIVE_INFINITY)
                  vv[k] = minmax.elementAt(i)[0];
                if (vv[k] == Double.POSITIVE_INFINITY)
                  vv[k] = minmax.elementAt(i)[1];
                y[k] = (int) Math.round(dy * (vv[k] - minmax.elementAt(i)[0]) / (minmax.elementAt(i)[1] - minmax.elementAt(i)[0]));
              }
              int hh=Math.max(1,y[1] - y[0]);
              if (isInteger) {
                int step = (int) Math.round(dy / (absMax - absMin + 1));
                if (step>1) {
                  y[0] = step * (int) Math.round(vv[0] - absMin);
                  hh = step * (int) Math.round(vv[1] - vv[0] + 1);
                  y[1] = y[0] + hh;
                }
              }
              g.fillRect(x + 2, offsetY + dy - y[1], barW + 1, hh);
              found = true;
            }
        }
      }
    }
    return image;
  }

}
