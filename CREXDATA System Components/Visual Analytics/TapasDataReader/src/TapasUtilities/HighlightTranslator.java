package TapasUtilities;

import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

/**
 * Translates its highlighting to another highlighter and vice versa.
 */
public class HighlightTranslator extends SingleHighlightManager
    implements ChangeListener {
  /**
   * Another highlighter with whom to synchronize highlighting.
   */
  public SingleHighlightManager otherHighlighter=null;
  /**
   * Keeps correspondence between objects in this highlighter and objects
   * in the other highlighter
   */
  public Translator translator=null;
  /**
   * Indicates whether a change in highlighting has been caused by a change
   * in the other highlighter.
   */
  protected boolean backTranslation=false;
  
  public void setOtherHighlighter(SingleHighlightManager otherHighlighter) {
    this.otherHighlighter = otherHighlighter;
    if (otherHighlighter!=null)
      otherHighlighter.addChangeListener(this);
  }

  public void notifyChange(){
    if (!backTranslation && otherHighlighter!=null && translator!=null) {
      otherHighlighter.removeChangeListener(this);
      if (highlighted == null)
        otherHighlighter.clearHighlighting();
      else
        otherHighlighter.highlight(translator.getSecond(highlighted));
      otherHighlighter.addChangeListener(this);
    }
    super.notifyChange();
  }

  public void setTranslator(Translator translator) {
    this.translator = translator;
  }
  
  public void stateChanged(ChangeEvent e) {
    if (e.getSource().equals(otherHighlighter) && translator!=null) {
      Object other=otherHighlighter.getHighlighted();
      backTranslation=true;
      if (other==null)
        clearHighlighting();
      else
        highlight(translator.getFirst(other));
      backTranslation=false;
    }
  }
}
