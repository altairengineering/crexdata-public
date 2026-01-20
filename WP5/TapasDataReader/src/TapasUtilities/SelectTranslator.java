package TapasUtilities;

import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.util.ArrayList;

/**
 * Translates its selections to another highlighter and vice versa.
 */
public class SelectTranslator extends ItemSelectionManager
    implements ChangeListener {
  /**
   * Another selector with whom to synchronize highlighting.
   */
  public ItemSelectionManager otherSelector=null;
  /**
   * Keeps correspondence between objects in this highlighter and objects
   * in the other highlighter
   */
  public Translator translator=null;
  /**
   * Indicates whether a change in selection has been caused by a change
   * in the other selector.
   */
  protected boolean backTranslation=false;
  
  public void setOtherSelector(ItemSelectionManager otherSelector) {
    this.otherSelector = otherSelector;
    if (otherSelector!=null)
      otherSelector.addChangeListener(this);
  }
  
  public void setTranslator(Translator translator) {
    this.translator = translator;
  }
  
  public void notifyChange(){
    if (!backTranslation && otherSelector!=null && translator!=null) {
      otherSelector.removeChangeListener(this);
      if (selected==null || selected.isEmpty())
        otherSelector.deselectAll();
      else {
        ArrayList others=new ArrayList(selected.size());
        for (Object s:selected) {
          Object other=translator.getSecond(s);
          if (other!=null)
            others.add(other);
        }
        otherSelector.updateSelection(others);
      }
      otherSelector.addChangeListener(this);
    }
    super.notifyChange();
  }
  
  public void stateChanged(ChangeEvent e) {
    if (e.getSource().equals(otherSelector) && translator!=null) {
      ArrayList others=otherSelector.getSelected();
      backTranslation=true;
      if (others==null || others.isEmpty())
        deselectAll();
      else {
        ArrayList mySelection=new ArrayList(others.size());
        for (Object s:others) {
          Object my=translator.getFirst(s);
          if (my!=null)
            mySelection.add(my);
        }
        updateSelection(mySelection);
      }
      backTranslation=false;
    }
  }
}
