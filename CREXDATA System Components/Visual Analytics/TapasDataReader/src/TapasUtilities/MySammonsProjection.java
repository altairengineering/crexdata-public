package TapasUtilities;

import TapasUtilities.gunther_foidl.SammonsProjection;

import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

public class MySammonsProjection extends SammonsProjection {
  /**
   * Projection copy with the minimal stress
   */
  public double bestProjection[][]=null;
  /**
   * The stress of the best projection
   */
  public double minStress=Double.NaN;
  public boolean done=false, mustStop=false;
  
  public MySammonsProjection(double[][] distanceMatrix,int outputDimension,int maxIterations,
                             boolean useEuclid) {
    super(distanceMatrix,null,outputDimension,maxIterations,useEuclid);
  }
  
  public void forceToStop() {
    mustStop=true;
  }
  
  public void runProjection(int nStepsBetweenNotifications,
                            int maxStepsWithNoImprovement,
                            ChangeListener listener) {
    if (this._distanceMatrix==null || listener==null)
      return;
    int nImprovementSteps=0, nStepsTotal=0;
    int i0=this.Iteration;
    System.out.println("Projection "+OutputDimension+"D starts");
    for (int i = this._maxIteration; i >= i0; i--) {
      if (mustStop)
        break;
      this.Iterate();
      if (mustStop)
        break;
      ++nImprovementSteps; ++nStepsTotal;
      double stress=computeStress();
      if (nStepsTotal % 50 == 0)
        System.out.println("Projection "+OutputDimension+"D: "+
                               nStepsTotal+" steps done; stress = "+stress+", min stress = "+minStress);
      if (bestProjection==null || stress<minStress) {
        bestProjection=makeProjectionCopy(Projection);
        minStress=stress;
        if (nImprovementSteps>=nStepsBetweenNotifications) {
          nImprovementSteps=0;
          if (mustStop)
            break;
          listener.stateChanged(new ChangeEvent(this));
          //System.out.println("Projection: notified the listener");
        }
      }
      else
        if (nImprovementSteps>=maxStepsWithNoImprovement) {
          System.out.println("Projection "+OutputDimension+"D: no improvement in "+
                                 nImprovementSteps+" steps; stopping the process");
          break;
        }
    }
    if (mustStop)
      return;
    done=true;
    if (bestProjection!=null)
      Projection=bestProjection;
    if (!mustStop)
      listener.stateChanged(new ChangeEvent(this));
    System.out.println("Projection "+OutputDimension+"D done; min stress = "+minStress);
  }
  
  public static double[][] makeProjectionCopy(double projection[][]) {
    if (projection==null)
      return null;
    double copy[][]=new double[projection.length][projection[0].length];
    for (int j=0; j<projection.length; j++)
      for (int k=0; k<projection[j].length; k++)
        copy[j][k]=projection[j][k];
    return copy;
  }
}
