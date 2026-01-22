using System.Collections;
using UnityEngine;
using MixedReality.Toolkit.SpatialManipulation;

public class DisablePanelManipulation : MonoBehaviour
{
    public GameObject panel;

    private ObjectManipulator objectManipulator;
    private MinMaxScaleConstraint minMaxScaleConstraint;
    private SolverHandler solverHandler;
    private TapToPlace tapToPlace;

    void Start()
    {
        if (panel != null)
        {
            objectManipulator = panel.GetComponent<ObjectManipulator>();
            minMaxScaleConstraint = panel.GetComponent<MinMaxScaleConstraint>();
            solverHandler = panel.GetComponent<SolverHandler>();
            tapToPlace = panel.GetComponent<TapToPlace>();
        }
    }

    public void DisableManipulation()
    {
        if (panel != null)
        {
            if (objectManipulator != null)
            {
                objectManipulator.enabled = false;
            }

            if (minMaxScaleConstraint != null)
            {
                minMaxScaleConstraint.enabled = false;
            }

            if (solverHandler != null)
            {
                solverHandler.enabled = false;
            }

            if (tapToPlace != null)
            {
                tapToPlace.enabled = false;
            }
        }
    }
}
