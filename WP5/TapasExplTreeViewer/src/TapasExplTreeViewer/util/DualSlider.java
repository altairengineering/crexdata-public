package TapasExplTreeViewer.util;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.ArrayList;

public class DualSlider extends JComponent {
  public static Color activeRangeColor=new Color(120, 180, 240);
  public static Color activeKnobColor=new Color(80, 120, 240);

  private int min, max, lowerValue, upperValue; //can be percentages or percentiles
  private int knobSize = 10;
  private boolean draggingLower = false, draggingUpper = false, draggingRange = false;
  private int lastMouseX = -1;
  private boolean enabled = true; // Indicates whether the slider is enabled

  private ArrayList<ChangeListener> listeners=null;

  public DualSlider(int min, int max, int lowerValue, int upperValue) {
    this.min = min;
    this.max = max;
    this.lowerValue = lowerValue;
    this.upperValue = upperValue;
    setPreferredSize(new Dimension(100, 20));

    addMouseListener(new MouseAdapter() {
      @Override
      public void mousePressed(MouseEvent e) {
        if (!enabled) return;
        handleMousePressed(e.getX());
      }

      @Override
      public void mouseReleased(MouseEvent e) {
        if (!enabled) return;
        draggingLower = draggingUpper = draggingRange = false;
        lastMouseX = -1;
      }

      @Override
      public void mouseClicked(MouseEvent e) {
        if (!enabled)
          return;
        if (e.getClickCount()==2 && isRangeLimited()) {
          resetLimitsToMinMax();
          draggingLower = draggingUpper = draggingRange = false;
          lastMouseX = -1;
          notifyChangeListeners();
        }
      }
    });

    addMouseMotionListener(new MouseAdapter() {
      @Override
      public void mouseDragged(MouseEvent e) {
        if (!enabled) return;
        handleMouseDragged(e.getX());
      }
    });
  }


  private void handleMousePressed(int mouseX) {
    int width = getWidth() - 2 * knobSize;
    int xLower = knobSize + (lowerValue - min) * width / (max - min);
    int xUpper = knobSize + (upperValue - min) * width / (max - min);

    if (xLower==xUpper) {
      if (xLower<=knobSize*2)
        draggingUpper=true;
      else if (xLower>=getWidth()-knobSize*2)
        draggingLower=true;
      else
        draggingRange=true;
    }
    else
    if (Math.abs(mouseX - xLower) <= knobSize / 2) {
      draggingLower = true;
    } else if (Math.abs(mouseX - xUpper) <= knobSize / 2) {
      draggingUpper = true;
    } else if (mouseX >= xLower && mouseX <= xUpper) {
      draggingRange = true;
    }
    lastMouseX = mouseX;
  }

  private void handleMouseDragged(int mouseX) {
    int width = getWidth() - 2 * knobSize;
    if (draggingLower) {
      lowerValue = min + (mouseX - knobSize) * (max - min) / width;
      lowerValue = Math.max(min, Math.min(upperValue, lowerValue));
    } else if (draggingUpper) {
      upperValue = min + (mouseX - knobSize) * (max - min) / width;
      upperValue = Math.max(lowerValue, Math.min(max, upperValue));
    } else if (draggingRange && lastMouseX != -1) {
      int delta = (mouseX - lastMouseX) * (max - min) / width;
      int newLowerValue = lowerValue + delta;
      int newUpperValue = upperValue + delta;

      if (newLowerValue >= min && newUpperValue <= max) {
        lowerValue = newLowerValue;
        upperValue = newUpperValue;
        lastMouseX = mouseX;
      }
    }
    repaint();
    notifyChangeListeners();
  }

  @Override
  protected void paintComponent(Graphics g) {
    super.paintComponent(g);

    int width = getWidth() - 2 * knobSize;
    int xLower = knobSize + (lowerValue - min) * width / (max - min);
    int xUpper = knobSize + (upperValue - min) * width / (max - min);

    // Determine colors based on enabled state
    Color trackColor = enabled ? Color.LIGHT_GRAY : Color.LIGHT_GRAY;
    Color rangeColor = enabled ? activeRangeColor : Color.GRAY;
    Color knobColor = enabled ? activeKnobColor : Color.GRAY;

    // Draw track
    g.setColor(trackColor);
    int trackLineWidth=knobSize-4, trackY=getHeight() / 2 - trackLineWidth/2;
    g.fillRect(knobSize, trackY, width, trackLineWidth);

    // Highlight range
    g.setColor(rangeColor);
    g.fillRect(xLower, trackY, xUpper - xLower, trackLineWidth);

    // Draw knobs
    g.setColor(knobColor);
    g.fillOval(xLower - knobSize / 2, getHeight() / 2 - knobSize / 2, knobSize, knobSize);
    g.fillOval(xUpper - knobSize / 2, getHeight() / 2 - knobSize / 2, knobSize, knobSize);
  }

  public int getMin() {
    return min;
  }

  public int getMax() {
    return max;
  }

  public int getLowerValue() {
    return lowerValue;
  }

  public void setLowerValue(int lowerValue) {
    if (lowerValue==this.lowerValue)
      return;
    this.lowerValue = lowerValue;
    repaint();
  }

  public int getUpperValue() {
    return upperValue;
  }

  public void setUpperValue(int upperValue) {
    if (upperValue==this.upperValue)
      return;
    this.upperValue = upperValue;
    repaint();
  }

  public void setEnabled(boolean enabled) {
    this.enabled = enabled;
    repaint();
  }

  public boolean isRangeLimited() {
    return lowerValue>min || upperValue<max;
  }

  public boolean resetLimitsToMinMax() {
    if (isRangeLimited()) {
      lowerValue=min; upperValue=max;
      repaint();
      return true;
    }
    return false;
  }

  public void addChangeListener(ChangeListener listener) {
    if (listeners==null)
      listeners=new ArrayList<ChangeListener>(10);
    if (!listeners.contains(listener))
      listeners.add(listener);
  }

  private void notifyChangeListeners() {
    if (listeners==null || listeners.isEmpty())
      return;
    ChangeEvent event = new ChangeEvent(this);
    for (ChangeListener listener : listeners) {
      listener.stateChanged(event);
    }
  }
}
