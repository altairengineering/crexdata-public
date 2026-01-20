package TapasUtilities;

import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.util.ArrayList;

public class ItemSelectionManager {
  public ArrayList selected=null;
  
  protected ArrayList<ChangeListener> changeListeners=null;
  
  public void addChangeListener(ChangeListener l) {
    if (changeListeners==null)
      changeListeners=new ArrayList(5);
    if (!changeListeners.contains(l))
      changeListeners.add(l);
  }
  
  public void removeChangeListener(ChangeListener l) {
    if (l!=null && changeListeners!=null)
      changeListeners.remove(l);
  }
  
  public void notifyChange(){
    if (changeListeners==null || changeListeners.isEmpty())
      return;
    ChangeEvent e=new ChangeEvent(this);
    for (ChangeListener l:changeListeners)
      l.stateChanged(e);
  }
  
  public void select(Object obj) {
    if (obj==null)
      return;
    if (selected==null)
      selected=new ArrayList(50);
    if (!selected.contains(obj)) {
      selected.add(obj);
      notifyChange();
    }
  }
  
  
  public void select(Object obj, boolean clearPreviousSelection) {
    if (!clearPreviousSelection || selected==null || selected.isEmpty()) {
      select(obj);
      return;
    }
    if (selected.size()==1 && selected.get(0).equals(obj))
      return;
    selected.clear();
    selected.add(obj);
    notifyChange();
  }
  
  public void select(ArrayList toSelect) {
    if (toSelect==null || toSelect.isEmpty())
      return;
    if (selected==null)
      selected=new ArrayList(50);
    boolean changed=false;
    for (Object s:toSelect)
      if (!selected.contains(s)) {
        selected.add(s);
        changed=true;
      }
    if (changed)
      notifyChange();
  }
  
  public void deselect(Object obj) {
    if (obj==null || selected==null || selected.isEmpty() || !selected.contains(obj))
      return;
    selected.remove(obj);
    notifyChange();
  }
  
  public void deselect(ArrayList objList) {
    if (objList==null || objList.isEmpty() ||
            selected==null || selected.isEmpty())
      return;
    boolean changed=false;
    for (int i=0; i<objList.size(); i++) {
      int idx=selected.indexOf(objList.get(i));
      if (idx>=0) {
        selected.remove(idx);
        changed=true;
      }
    }
    if (changed)
      notifyChange();
  }
  
  public void deselectAll() {
    if (selected==null || selected.isEmpty())
      return;
    selected.clear();
    notifyChange();
  }
  
  public boolean hasSelection() {
    return selected!=null && !selected.isEmpty();
  }
  
  public boolean isSelected(Object obj) {
    return obj!=null && selected!=null && selected.contains(obj);
  }
  
  public boolean areAllSelected(ArrayList objList) {
    if (objList==null || objList.isEmpty() ||
            selected==null || selected.isEmpty() || selected.size()<objList.size())
      return false;
    for (int i=0; i<objList.size(); i++)
      if (!selected.contains(objList.get(i)))
        return false;
    return true;
  }
  
  public int indexOf(Object obj) {
    if (selected!=null && obj!=null)
      return selected.indexOf(obj);
    return -1;
  }
  
  public ArrayList getSelected(){
    return selected;
  }
  
  public void updateSelection(ArrayList newSelection) {
    if (selected!=null)
      selected.clear();
    if (newSelection!=null) {
      if (selected==null)
        selected=new ArrayList(50);
      for (int i = 0; i < newSelection.size(); i++)
        selected.add(newSelection.get(i));
    }
    notifyChange();
  }
  
  /**
   * Assumes that both lists contain unique elements, no duplicates
   */
  public static boolean sameContent(ArrayList a1, ArrayList a2) {
    if (a1==null || a1.isEmpty())
      return a2==null || a2.isEmpty();
    if (a2==null || a2.isEmpty())
      return false;
    if (a1.size()!=a2.size())
      return false;
    for (int i=0; i<a1.size(); i++)
      if (!a2.contains(a1))
        return false;
    return true;
  }
}
