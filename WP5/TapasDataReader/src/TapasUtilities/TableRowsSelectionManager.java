package TapasUtilities;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.event.ListSelectionEvent;
import javax.swing.event.ListSelectionListener;
import java.awt.*;
import java.util.ArrayList;

/**
 * Listens to highlighting and selection change events and responds by
 * highlighting and selection of table rows.
 */
public class TableRowsSelectionManager
    implements ChangeListener, ListSelectionListener {
  /**
   * The table where to highlight and select rows
   */
  public JTable table=null;
  protected SingleHighlightManager highlighter=null;
  protected ItemSelectionManager selector=null;
  protected int hlIdx=-1;
  
  public boolean mayScrollTable =true;
  
  public void setTable(JTable table) {
    this.table=table;
    if (table!=null)
      table.getSelectionModel().addListSelectionListener(this);
  }
  
  public void setHighlighter(SingleHighlightManager highlighter) {
    this.highlighter = highlighter;
    if (highlighter!=null)
      highlighter.addChangeListener(this);
  }
  
  public void setSelector(ItemSelectionManager selector) {
    this.selector = selector;
    if (selector!=null)
      selector.addChangeListener(this);
  }
  
  public void setMayScrollTable(boolean mayScrollTable) {
    this.mayScrollTable = mayScrollTable;
  }
  
  public void stateChanged(ChangeEvent e) {
    if (table==null)
      return;
    if (e.getSource().equals(selector)) {
      ArrayList selList=selector.getSelected();
      int rows[]=table.getSelectedRows();
      if (selList==null || selList.isEmpty()) {
        if (rows!=null)
          table.getSelectionModel().clearSelection();
      }
      else {
        int sel[]=new int[selList.size()];
        for (int i=0; i<selList.size(); i++)
          sel[i] = table.convertRowIndexToView((Integer) selList.get(i));
        if (rows!=null && rows.length==sel.length) {
          boolean same=true;
          for (int i=0; i<selList.size() && same; i++) {
            int idx=-1;
            for (int j = 0; j < rows.length && idx<0; j++)
              if (rows[j] == sel[i])
                idx=j;
            same=idx>=0;
          }
          if (same)
            return;
        }
        table.getSelectionModel().removeListSelectionListener(this);
        table.getSelectionModel().setSelectionInterval(sel[0],sel[0]);
        int min=sel[0];
        for (int i=1; i<sel.length; i++) {
          table.getSelectionModel().addSelectionInterval(sel[i], sel[i]);
          if (min>sel[i]) min=sel[i];
        }
        table.getSelectionModel().addListSelectionListener(this);
        if (mayScrollTable) {
          Rectangle rect = table.getCellRect(min, 0, true);
          table.scrollRectToVisible(rect);
        }
      }
    }
    else
      if (e.getSource().equals(highlighter)) {
        if (hlIdx>=0) {
          int row=table.convertRowIndexToView(hlIdx);
          Rectangle rect=table.getCellRect(row,0,true);
          rect.width=table.getWidth();
          table.repaint(rect);
        }
        Integer hl=(Integer)highlighter.getHighlighted();
        if (hl==null) return;
        hlIdx=hl;
        int row=table.convertRowIndexToView(hlIdx);
        Rectangle rect=table.getCellRect(row,0,true);
        rect.width=table.getWidth();
        table.repaint(rect);
        if (mayScrollTable)
          table.scrollRectToVisible(rect);
      }
  }
  public void valueChanged(ListSelectionEvent e) {
    table.getSelectionModel().removeListSelectionListener(this);
    int rows[]=table.getSelectedRows();
    if (rows==null || rows.length<1)
      selector.deselectAll();
    else {
      ArrayList<Integer> selection=new ArrayList<Integer>(rows.length);
      for (int i=0; i<rows.length; i++)
        selection.add(table.convertRowIndexToModel(rows[i]));
      selector.updateSelection(selection);
    }
    table.getSelectionModel().addListSelectionListener(this);
  }
}
