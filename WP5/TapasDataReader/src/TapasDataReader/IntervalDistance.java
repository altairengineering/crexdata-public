package TapasDataReader;

public class IntervalDistance {
  public static double distance(double a1, double a2, double b1, double b2) {
    if (Double.isNaN(a1) || Double.isInfinite(a1)) a1=Integer.MIN_VALUE;
    if (Double.isNaN(a2) || Double.isInfinite(a2)) a2=Integer.MAX_VALUE;
    if (Double.isNaN(b1) || Double.isInfinite(b1)) b1=Integer.MIN_VALUE;
    if (Double.isNaN(b2) || Double.isInfinite(b2)) b2=Integer.MAX_VALUE;
    double da1b1=Math.abs(Math.min(a1,a2)-Math.min(b1,b2)),
           da2b2=Math.abs(Math.max(a1,a2)-Math.max(b1,b2));
    return (da1b1+da2b2)/2;
  }
  
  public static double distanceRelative(double a1, double a2, double b1, double b2,
                                        double absMin, double absMax) {
    if (Double.isNaN(absMin) || Double.isNaN(absMax) ||
            Double.isInfinite(absMin) || Double.isInfinite(absMax))
      return distance(a1,a2,b1,b2);
    if (Double.isNaN(a1) || Double.isInfinite(a1)) a1=absMin;
    if (Double.isNaN(a2) || Double.isInfinite(a2)) a2=absMax;
    if (Double.isNaN(b1) || Double.isInfinite(b1)) b1=absMin;
    if (Double.isNaN(b2) || Double.isInfinite(b2)) b2=absMax;
    double da1b1=Math.abs(Math.min(a1,a2)-Math.min(b1,b2)),
        da2b2=Math.abs(Math.max(a1,a2)-Math.max(b1,b2));
    return (da1b1+da2b2)/2/(absMax-absMin);
  }
}
