using System.Collections.Generic;
using UnityEngine;

public class QRCodeManager : MonoBehaviour
{
    // Assign this in the Inspector: the object to move each time a saved QR code is detected in Phase 2.
    public GameObject objectToMove;

    // List to keep track of saved QR codes.
    private List<string> savedQRCodes = new List<string>();

    // Enum to track the current phase of the system.
    private enum Phase { Phase1, Phase2 }
    private Phase currentPhase = Phase.Phase1;

    /// <summary>
    /// Call this method when a QR code is detected.
    /// In Phase 1, it saves new QR codes and prints to the console.
    /// In Phase 2, if the QR code was already saved, it prints a different message and moves the object.
    /// </summary>
    /// <param name="qrData">The string data from the detected QR code.</param>
    public void OnQRCodeDetected(string qrData)
    {
        if (currentPhase == Phase.Phase1)
        {
            if (!savedQRCodes.Contains(qrData))
            {
                savedQRCodes.Add(qrData);
                Debug.Log("Phase 1: New QR code detected and saved: " + qrData);
            }
            else
            {
                // Optionally handle duplicate detections in Phase 1
                Debug.Log("Phase 1: QR code already seen: " + qrData);
            }
        }
        else if (currentPhase == Phase.Phase2)
        {
            if (savedQRCodes.Contains(qrData))
            {
                Debug.Log("Phase 2: Saved QR code detected: " + qrData);
                if (objectToMove != null)
                {
                    // Move the object one unit to the left.
                    objectToMove.transform.position += Vector3.left;
                }
                else
                {
                    Debug.LogWarning("objectToMove is not assigned in the inspector.");
                }
            }
            else
            {
                // If a new QR code is detected in Phase 2, you might ignore it or handle it differently.
                Debug.Log("Phase 2: Unrecognized QR code detected (ignored): " + qrData);
            }
        }
    }

    /// <summary>
    /// Call this method (e.g., via a button press) to switch the system from Phase 1 to Phase 2.
    /// </summary>
    public void SwitchToPhaseTwo()
    {
        currentPhase = Phase.Phase2;
        Debug.Log("Switched to Phase 2: Only saved QR codes will be processed.");
    }
}
