package TapasExplTreeViewer.vis;

import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;
import java.util.List;

public class MultiHeatmapPanel extends JPanel {
  private List<HeatmapDrawer> heatmaps = new ArrayList<HeatmapDrawer>();
  private List<String> titles = new ArrayList<String>();
  private int padding = 5, titleHeight=20; // Space between heatmaps and space for title

  public MultiHeatmapPanel() {
    setLayout(null); // Use absolute positioning to allow custom layout
    ToolTipManager.sharedInstance().registerComponent(this);
    ToolTipManager.sharedInstance().setDismissDelay(Integer.MAX_VALUE);
  }

  /**
   * Adds a heatmap to the panel.
   *
   * @param heatmap The HeatmapDrawer instance.
   * @param title   The title to display above the heatmap.
   */
  public void addHeatmap(HeatmapDrawer heatmap, String title) {
    heatmaps.add(heatmap);
    titles.add(title);
    add(heatmap);
    repaint();
  }

  public void setHeatmapTitle(String title, int idx) {
    if (titles!=null && idx>=0 && idx<titles.size())
      titles.set(idx,title);
  }

  @Override
  protected void paintComponent(Graphics g) {
    super.paintComponent(g);

    if (heatmaps.isEmpty()) return;

    int panelWidth = getWidth();
    int panelHeight = getHeight();

    // Calculate the number of columns
    Dimension minSize=heatmaps.get(0).getMinumumSize();
    int maxNumRows = Math.max(1, panelHeight / (minSize.height + titleHeight+padding));
    int numCols = (int) Math.ceil((double) heatmaps.size() / maxNumRows);
    int numRows= (int) Math.ceil((double) heatmaps.size() / numCols);

    int cellWidth = (panelWidth - (numCols + 1) * padding) / numCols;
    while (cellWidth>2.5*minSize.width) {
      ++numCols;
      cellWidth = (panelWidth - (numCols + 1) * padding) / numCols;
      numRows= (int) Math.ceil((double) heatmaps.size() / numCols);
    }

    int nEmptyCells=numRows*numCols-heatmaps.size();
    while (nEmptyCells>=numCols/2) {
      int nr=numRows-1, nc=(int) Math.ceil((double) heatmaps.size() / nr);
      if (nr*nc-heatmaps.size()<nEmptyCells && (panelWidth - (nc + 1) * padding) / nc >= minSize.width) {
        numRows=nr; numCols=nc;
        cellWidth = (panelWidth - (numCols + 1) * padding) / numCols;
      }
      else {
        --nc;
        nr=(int) Math.ceil((double) heatmaps.size() / nc);
        if (nr*nc-heatmaps.size()<nEmptyCells && (panelHeight - (nr + 1) * padding) / nr >= minSize.height) {
          numRows=nr; numCols=nc;
          cellWidth = (panelWidth - (numCols + 1) * padding) / numCols;
        }
        else
          break;
      }
      nEmptyCells=numRows*numCols-heatmaps.size();
    }

    int cellHeight = (panelHeight - (numRows + 1) * padding) / numRows;

    // Position heatmaps
    for (int i = 0; i < heatmaps.size(); i++) {
      int row = i / numCols;
      int col = i % numCols;

      int x = padding + col * (cellWidth + padding);
      int y = padding + row * (cellHeight + padding);

      HeatmapDrawer heatmap = heatmaps.get(i);
      heatmap.setSize(cellWidth,cellHeight);
      heatmap.setBounds(x, y + titleHeight, cellWidth, cellHeight - titleHeight); // Leave space for title

      // Draw the title above the heatmap
      String title = titles.get(i);
      g.setColor(Color.BLACK);
      g.setFont(new Font("SansSerif", Font.BOLD, 12));
      FontMetrics fm = g.getFontMetrics();
      int titleWidth = fm.stringWidth(title);
      g.drawString(title, x + (cellWidth - titleWidth) / 2, y + fm.getAscent());
    }
  }

  @Override
  public void doLayout() {
    // Trigger layout when panel is resized
    revalidate();
    repaint();
  }
}
