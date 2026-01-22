using Microsoft.MixedReality.OpenXR;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.ARSubsystems;

/// <summary>
/// Allows multiple QR codes (ARMarkers) to be calibrated, then in operation mode the system 
/// will dynamically switch to another visible, calibrated code if the current code is lost.
/// </summary>
public class CalibrationAndOperationOpenXR : MonoBehaviour
{
    [Header("References")]
    [Tooltip("Parent of all your holograms. Moving this reposition the entire scene.")]
    public Transform environmentRoot;

    [Tooltip("A prefab for the calibration button to spawn for each detected marker.")]
    public GameObject calibrationButtonPrefab;

    [Tooltip("Distance (in meters) to place the calibration button in front of each marker.")]
    public float buttonDistance = 0.2f;

    /// <summary>Whether we are in calibration mode (true) or operation mode (false).</summary>
    private bool isCalibrationMode = true;

    // A dictionary storing each marker’s offset from environmentRoot. Key = TrackableId
    private Dictionary<TrackableId, CalibrationOffset> calibrationOffsets
        = new Dictionary<TrackableId, CalibrationOffset>();

    // Represents the environmentRoot’s offset (position + rotation) relative to a particular code.
    private struct CalibrationOffset
    {
        public Vector3 positionOffset;
        public Quaternion rotationOffset;
    }

    // Manager that detects ARMarkers (QR codes)
    private ARMarkerManager markerManager;

    // The code we are “currently” using in operation mode. We switch if it's lost or invalid.
    private TrackableId currentUsedCode = TrackableId.invalidId;

    // A dictionary to track the last known position of each marker
    private Dictionary<TrackableId, Vector3> lastKnownPositions = new Dictionary<TrackableId, Vector3>();

    private float moveThreshold = 0.001f; // How far (in meters) a marker must move before we consider it "changed"




    [Header("Smooth Movement")]
    [Tooltip("Speed of smooth alignment. Higher = faster movement.")]
    public float alignmentSpeed = 5f;

    [Tooltip("Distance threshold to stop smoothing (prevents infinite micro-movements).")]
    public float positionThreshold = 0.001f;

    [Tooltip("Angle threshold to stop smoothing (in degrees).")]
    public float rotationThreshold = 0.1f;

    // Add these private fields for smooth movement
    private Vector3 targetPosition;
    private Quaternion targetRotation;
    private bool isSmoothing = false;

    private void Start()
    {
        markerManager = ARMarkerManager.Instance;
        if (markerManager == null)
        {
            Debug.LogError("[CalibOp] ARMarkerManager not found! Ensure an ARMarkerManager is in the scene.");
        }
        else
        {
            Debug.Log("[CalibOp] Started in CALIBRATION mode. You can calibrate multiple codes by pressing each code's button.");
        }
    }

    private void Update()
    {
        if (markerManager == null) { return; }

        // Handle smooth movement first
        if (isSmoothing && environmentRoot != null)
        {
            SmoothMoveToTarget();
        }

        if (isCalibrationMode)
        {
            //            Debug.Log("[CalibOp] (CALIBRATION) Checking for markers to spawn buttons...");

            // For each tracked marker, if it has no child button, spawn a calibration button
            foreach (var marker in markerManager.trackables)
            {
                if (marker == null) { continue; }

                // Only spawn a button if none exists yet
                if (marker.transform.childCount == 0 && calibrationButtonPrefab != null)
                {
                    SpawnCalibrationButton(marker);
                }
            }
        }
        else
        {

            //Debug.Log("[CalibOp] (OPERATION) Checking if any calibrated marker has changed position...");

            // Iterate over each calibrated code in your dictionary
            foreach (var kvp in calibrationOffsets)
            {
                TrackableId codeId = kvp.Key;
                // Find the corresponding ARMarker in markerManager
                ARMarker marker = FindMarkerById(codeId);
                if (marker == null)
                {
                    // Marker might not exist or might have been removed
                    continue;
                }

                // Get the current marker position
                Vector3 currentPos = marker.transform.position;

                // If we haven't recorded this marker's position before, store it and continue
                if (!lastKnownPositions.ContainsKey(codeId))
                {
                    lastKnownPositions[codeId] = currentPos;
                    continue;
                }

                // Compare current position to last known
                Vector3 oldPos = lastKnownPositions[codeId];
                float distance = Vector3.Distance(oldPos, currentPos);

                // If marker moved more than some tiny threshold, realign
                if (distance > moveThreshold)
                {
                    //Debug.Log($"[CalibOp] Marker {codeId} moved ({distance:F4}m). Re-aligning environmentRoot...");
                    AlignWithCode(codeId);

                    // Optionally, break if you only want to align once per frame:
                    // break;
                }

                // Update the last known position
                lastKnownPositions[codeId] = currentPos;

            }
        }
    }

    private void SmoothMoveToTarget()
    {
        Vector3 currentPos = environmentRoot.position;
        Quaternion currentRot = environmentRoot.rotation;

        // Lerp position
        Vector3 newPos = Vector3.Lerp(currentPos, targetPosition, alignmentSpeed * Time.deltaTime);

        // Slerp rotation
        Quaternion newRot = Quaternion.Slerp(currentRot, targetRotation, alignmentSpeed * Time.deltaTime);

        environmentRoot.SetPositionAndRotation(newPos, newRot);

        // Check if we're close enough to stop smoothing
        float posDistance = Vector3.Distance(newPos, targetPosition);
        float rotAngle = Quaternion.Angle(newRot, targetRotation);

        if (posDistance < positionThreshold && rotAngle < rotationThreshold)
        {
            // Snap to final position and stop smoothing
            environmentRoot.SetPositionAndRotation(targetPosition, targetRotation);
            isSmoothing = false;
            Debug.Log($"[CalibOp] Smooth alignment completed.");
        }
    }
    private ARMarker FindMarkerById(TrackableId codeId)
    {
        foreach (var marker in markerManager.trackables)
        {
            if (marker != null && marker.trackableId == codeId)
            {
                return marker;
            }
        }
        return null;
    }

    /// <summary>
    /// Creates a calibration button as a child of the marker so it moves with that marker.
    /// </summary>
    private void SpawnCalibrationButton(ARMarker marker)
    {
        Debug.Log($"[CalibOp] Spawning calibration button for marker {marker.trackableId}");

        GameObject buttonObj = Instantiate(calibrationButtonPrefab, marker.transform);
        buttonObj.transform.localPosition = new Vector3(0f, 0f, buttonDistance);
        buttonObj.transform.localRotation = Quaternion.identity;

        var cb = buttonObj.GetComponent<CalibrationButton>();
        if (cb != null)
        {
            cb.SetMarkerAndManager(marker, this);
        }
        else
        {
            Debug.LogWarning("[CalibOp] The calibration button prefab is missing the 'CalibrationButton' component!");
        }
    }

    /// <summary>
    /// Called by a calibration button script when the user presses the button to calibrate that marker.
    /// </summary>
    public void CalibrateMarker(ARMarker marker)
    {
        if (!isCalibrationMode)
        {
            Debug.LogWarning("[CalibOp] CalibrateMarker called but we're not in calibration mode!");
            return;
        }
        if (marker == null || environmentRoot == null)
        {
            Debug.LogError("[CalibOp] CalibrateMarker: marker or environmentRoot is null!");
            return;
        }

        TrackableId id = marker.trackableId;
        Debug.Log($"[CalibOp] CalibrateMarker called for code {id}.");

        // Compute offset: environmentRoot’s position/rotation in marker’s local coordinate space
        Vector3 posOffset = marker.transform.InverseTransformPoint(environmentRoot.position);
        Quaternion rotOffset = Quaternion.Inverse(marker.transform.rotation) * environmentRoot.rotation;

        CalibrationOffset offset = new CalibrationOffset
        {
            positionOffset = posOffset,
            rotationOffset = rotOffset
        };

        calibrationOffsets[id] = offset;
        Debug.Log($"[CalibOp] --> Code {id} CALIBRATED. positionOffset={posOffset}, rotationOffset={rotOffset}");

        // Show dictionary contents for debugging
        Debug.Log("[CalibOp] Current calibration dictionary:");
        foreach (var kvp in calibrationOffsets)
        {
            Debug.Log($"   Key={kvp.Key}, posOffset={kvp.Value.positionOffset}, rotOffset={kvp.Value.rotationOffset}");
        }
    }

    /// <summary>
    /// Called (e.g. via UI) after calibrating all desired markers. Removes buttons and switches to operation mode.
    /// </summary>
    public void FinishCalibration()
    {
        if (!isCalibrationMode)
        {
            Debug.LogWarning("[CalibOp] FinishCalibration called, but we're already in operation mode!");
            return;
        }

        Debug.Log("[CalibOp] FinishCalibration: removing calibration buttons and switching to operation mode.");

        // Remove all child objects (buttons) from each marker
        foreach (var marker in markerManager.trackables)
        {
            if (marker == null) { continue; }

            for (int i = marker.transform.childCount - 1; i >= 0; i--)
            {
                Destroy(marker.transform.GetChild(i).gameObject);
            }
        }

        // Switch to operation mode
        isCalibrationMode = false;
        // We haven't chosen any code yet, so let's reset currentUsedCode
        currentUsedCode = TrackableId.invalidId;

        Debug.Log("[CalibOp] --> Now in OPERATION mode! We'll realign environmentRoot to any visible calibrated code.");
    }

    /// <summary>
    /// Returns true if the marker with the given codeId is actively tracked in 'TrackingState.Tracking'.
    /// Also checks that the code is in our dictionary (i.e., calibrated).
    /// </summary>
    private bool IsMarkerVisible(TrackableId codeId)
    {
        if (codeId == TrackableId.invalidId) { return false; }
        if (!calibrationOffsets.ContainsKey(codeId))
        {
            // Not calibrated => "invisible" for our usage
            return false;
        }

        foreach (var marker in markerManager.trackables)
        {
            if (marker != null && marker.trackableId == codeId)
            {
                // Check the ARMarker's tracking state
                if (marker.trackingState == TrackingState.Tracking)
                {
                    // Actively tracked by the camera => truly visible
                    return true;
                }
                else
                {
                    Debug.Log($"[CalibOp] IsMarkerVisible({codeId}) => trackingState={marker.trackingState}");
                }
            }
        }
        return false;
    }

    /// <summary>
    /// Finds any calibrated code that is actively tracked in 'TrackingState.Tracking'.
    /// Returns invalidId if none is found.
    /// </summary>
    private TrackableId FindAnyVisibleCalibratedCode()
    {
        foreach (var marker in markerManager.trackables)
        {
            if (marker == null) { continue; }
            TrackableId id = marker.trackableId;

            // Confirm we have an offset for this code
            if (calibrationOffsets.ContainsKey(id))
            {
                // Check if it's actively tracked
                if (marker.trackingState == TrackingState.Tracking)
                {
                    Debug.Log($"[CalibOp] Found a visible calibrated code: {id}");
                    return id;
                }
                else
                {
                    Debug.Log($"[CalibOp] Code {id} is in dictionary but trackingState={marker.trackingState}");
                }
            }
        }
        return TrackableId.invalidId;
    }

    /// <summary>
    /// Aligns environmentRoot using the offset for the given codeId (if valid & currently tracked).
    /// </summary>
    private void AlignWithCode(TrackableId codeId)
    {
        if (!calibrationOffsets.TryGetValue(codeId, out CalibrationOffset offset))
        {
            Debug.LogWarning($"[CalibOp] AlignWithCode({codeId}) => code not in dictionary!");
            return;
        }

        // Find the actual ARMarker in the trackables
        ARMarker foundMarker = null;
        foreach (var marker in markerManager.trackables)
        {
            if (marker != null && marker.trackableId == codeId)
            {
                foundMarker = marker;
                break;
            }
        }
        if (foundMarker == null)
        {
            Debug.LogWarning($"[CalibOp] AlignWithCode({codeId}) => marker not found in trackables!");
            return;
        }

        // Compute the target position/rotation
        targetPosition = foundMarker.transform.TransformPoint(offset.positionOffset);
        targetRotation = foundMarker.transform.rotation * offset.rotationOffset;

        // Start smooth movement
        isSmoothing = true;

        //Debug.Log($"[CalibOp] Starting smooth alignment to code {codeId}");
    }
}
