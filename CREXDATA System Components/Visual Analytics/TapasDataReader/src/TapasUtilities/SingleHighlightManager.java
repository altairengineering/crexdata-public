package TapasUtilities;

import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.util.ArrayList;

public class SingleHighlightManager {
  public Object highlighted=null;
  
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
  
  public void highlight(Object obj) {
    if (obj==null)
      if (highlighted==null)
        return;
      else;
    else
      if (obj.equals(highlighted))
        return;
    highlighted=obj;
    notifyChange();
  }
  
  public void clearHighlighting() {
    if (highlighted==null)
      return;
    highlighted=null;
    notifyChange();
  }
  
  public Object getHighlighted(){
    return highlighted;
  }
}
