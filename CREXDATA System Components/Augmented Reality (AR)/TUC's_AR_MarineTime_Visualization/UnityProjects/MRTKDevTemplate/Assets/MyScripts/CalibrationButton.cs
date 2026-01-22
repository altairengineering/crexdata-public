using Microsoft.MixedReality.OpenXR;
using UnityEngine;

public class CalibrationButton : MonoBehaviour
{
    private ARMarker associatedMarker;
    private CalibrationAndOperationOpenXR manager;

    /// <summary>
    /// Called by the manager after instantiating this prefab to link the marker and the manager.
    /// </summary>
    public void SetMarkerAndManager(ARMarker marker, CalibrationAndOperationOpenXR manager)
    {
        associatedMarker = marker;
        this.manager = manager;
    }

    /// <summary>
    /// Called by a UI event, pressable button event, etc., when the user taps the button.
    /// </summary>
    public void OnButtonPressed()
    {
        if (associatedMarker != null && manager != null)
        {
            Debug.Log($"[CalibrationButton] Button pressed for marker {associatedMarker.trackableId}.");
            manager.CalibrateMarker(associatedMarker);
        }
        else
        {
            Debug.LogWarning("[CalibrationButton] Missing references! Cannot calibrate marker.");
        }
    }
}
