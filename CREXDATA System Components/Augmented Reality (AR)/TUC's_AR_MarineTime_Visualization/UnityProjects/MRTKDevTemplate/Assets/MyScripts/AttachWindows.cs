using UnityEngine;

public class MaintainConstantGapUsingBounds : MonoBehaviour
{
    [SerializeField] public GameObject primaryWindow;
    [SerializeField] public GameObject secondaryWindow;
    [SerializeField] public float fixedGap = 0.1f;

    private Renderer[] primaryRenderers;
    private Vector3 initialOffset;         // Initial difference in position (world space)
    private Quaternion initialRotOffset;   // Initial difference in rotation

    private void Start()
    {
        if (primaryWindow == null || secondaryWindow == null)
        {
            Debug.LogError("Assign both windows in the inspector!");
            enabled = false;
            return;
        }

        // Get all renderers for bounding-box calculations
        primaryRenderers = primaryWindow.GetComponentsInChildren<Renderer>();

        // Store the initial position offset (in world space)
        initialOffset = secondaryWindow.transform.position - primaryWindow.transform.position;

        // Store the initial rotation offset:
        // "How much is secondaryWindow rotated relative to primaryWindow?"
        initialRotOffset = Quaternion.Inverse(primaryWindow.transform.rotation) 
                           * secondaryWindow.transform.rotation;
    }

    private void Update()
{
    if (primaryWindow == null || secondaryWindow == null) return;
    if (primaryRenderers == null || primaryRenderers.Length == 0) return;

    // 1) Combine bounds of the primaryWindow
    Bounds combinedBounds = GetCombinedBounds(primaryRenderers);

    // 2) The top edge of the primary window in world space
    float topY = combinedBounds.max.y;

    // 3) Start with the "initial offset" for x and z
    Vector3 newPos = primaryWindow.transform.position + initialOffset;

    // 4) Overwrite only the Y so it's always topY + gap
    newPos.y = topY + fixedGap;

    // Set the new position of the secondary window
    secondaryWindow.transform.position = newPos;

    // -- ROTATE THE SECONDARY WINDOW TO FACE THE CAMERA --
    Camera mainCamera = Camera.main;
    if (mainCamera != null)
    {
        Vector3 lookTarget = mainCamera.transform.position;
        lookTarget.y = secondaryWindow.transform.position.y; // maintain upright orientation
        
        secondaryWindow.transform.LookAt(lookTarget);

        // Rotate 180 degrees around the Y-axis to flip the window
        secondaryWindow.transform.Rotate(0, 180f, 0);
}

}


    private Bounds GetCombinedBounds(Renderer[] renderers)
    {
        Bounds combined = new Bounds();
        bool first = true;
        foreach (Renderer r in renderers)
        {
            if (!r) continue;
            if (first)
            {
                combined = r.bounds;
                first = false;
            }
            else
            {
                combined.Encapsulate(r.bounds);
            }
        }
        return combined;
    }
}
