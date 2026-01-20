package TapasExplTreeViewer.vis;

import java.awt.*;

public class MyColors {
  public static Color niceColors[]={
      new Color(0,127,255),
      new Color(255,0,127),
      new Color(255,127,0),
      new Color(127,0,255),
      new Color(166, 206, 227), // "#a6cee3"
      new Color(31, 120, 180),  // "#1f78b4"
      new Color(178, 223, 138), // "#b2df8a"
      new Color(51, 160, 44),   // "#33a02c"
      new Color(251, 154, 153), // "#fb9a99"
      new Color(227, 26, 28),   // "#e31a1c"
      new Color(253, 191, 111), // "#fdbf6f"
      new Color(255, 127, 0),   // "#ff7f00"
      new Color(178, 106, 214), // "#b26ad6"
      new Color(106, 61, 154),  // "#6a3d9a"
      new Color(255, 255, 153), // "#ffff99"
      new Color(177, 89, 40),   // "#b15928"
      new Color(234, 85, 69),   // "#ea5545"
      new Color(244, 106, 155), // "#f46a9b"
      new Color(239, 155, 32),  // "#ef9b20"
      new Color(237, 191, 51),  // "#edbf33"
      new Color(237, 225, 91),  // "#ede15b"
      new Color(189, 207, 50),  // "#bdcf32"
      new Color(135, 188, 69),  // "#87bc45"
      new Color(39, 174, 239),  // "#27aeef"
      new Color(179, 61, 198)  // "#b33dc6"
  };

  public static Color getNiceColor (int idx) {
    if (idx<0) return Color.white;
    if (idx>=niceColors.length) return Color.black;
    return niceColors[idx];
  }

  public static Color getNiceColorExt (int idx) {
    if (idx<0) return Color.white;
    if (idx<niceColors.length)
      return niceColors[idx];
    idx-=niceColors.length;
    if (idx<niceColors.length)
      return niceColors[idx].darker();
    idx-=niceColors.length;
    if (idx<niceColors.length) {
      float hsb[]=Color.RGBtoHSB(niceColors[idx].getRed(),niceColors[idx].getGreen(),niceColors[idx].getBlue(),null);
      return Color.getHSBColor(hsb[0],hsb[1]*0.6f,hsb[2]);
    }
    return Color.getHSBColor((float)Math.random(),
        (float)Math.max(Math.random(),0.5),
        (float)Math.max(Math.random(),0.5));
  }

  public static Color getTextColorBasedOnBrightness(Color backgroundColor) {
    // Calculate the luminance of the background color
    double luminance = 0.299 * backgroundColor.getRed() +
        0.587 * backgroundColor.getGreen() +
        0.114 * backgroundColor.getBlue();

    // Return black text on light backgrounds and white text on dark backgrounds
    return luminance > 128 ? Color.BLACK : Color.WHITE;
  }
}
