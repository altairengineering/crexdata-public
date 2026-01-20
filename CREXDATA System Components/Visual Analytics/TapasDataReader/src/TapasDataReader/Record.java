package TapasDataReader;

public class Record {
  public String sector, flight, FromS, ToS, FromT, ToT;
  public int step, delay, FromN, ToN;

  protected int calc(String s) {
    int n=Integer.valueOf(s.substring(0,2)).intValue()*60+Integer.valueOf(s.substring(3,5)).intValue();
    return n;
  }
  public void calc() {
    FromN=calc(FromT);
    ToN=calc(ToT);
  }

  public Record clone() {
    Record r=new Record();
    r.sector=sector;
    r.flight=flight;
    r.FromS=FromS;
    r.ToS=ToS;
    r.FromT=FromT;
    r.ToT=ToT;
    r.step=step;
    r.delay=delay;
    r.FromN=FromN;
    r.ToN=ToN;
    return r;
  }

}
