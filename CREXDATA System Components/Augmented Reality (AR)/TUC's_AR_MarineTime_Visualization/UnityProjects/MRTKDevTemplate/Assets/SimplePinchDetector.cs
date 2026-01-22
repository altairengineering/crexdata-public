using UnityEngine;
using System.Collections.Generic;
using UnityEngine.UI;
using MixedReality.Toolkit.UX;
using MixedReality.Toolkit.Input; // MRTK3 input

// Alias for Unity UI Slider
using UnityUISlider = UnityEngine.UI.Slider;

public class SimplePinchDetector : MonoBehaviour
{
    [Header("Corner Settings")]
    public int totalCorners = 4;

    [Header("Prefabs (Optional)")]
    public GameObject previewMarkerPrefab;
    public GameObject confirmedCornerMarkerPrefab;

    [Header("Pinch Gesture Settings")]
    [Tooltip("Time (in seconds) the user must hold pinch to confirm a corner.")]
    public float pinchHoldTime = 2.0f;
    [Range(0.1f, 0.9f)]
    public float pinchThreshold = 0.5f;

    [Header("Hand Tracking")]
    [Tooltip("Assign the hand controller (left or right)")]
    public ArticulatedHandController handController;

    [Header("Mesh Material")]
    public Material panelMaterial;

    [Header("Alternate Material (Optional)")]
    public Material alternatePanelMaterial;

    [Header("Delete Button")]
    public GameObject deleteButtonPrefab;

    /// <summary>
    /// Whether panel-making is currently enabled.
    /// If false, corners can't be confirmed, and the preview marker is hidden.
    /// </summary>
    [Header("Panel Making Activation")]
    public bool panelMakingActive = true;

    private Vector3[] corners;
    private int currentCornerIndex = 0;
    private bool isPinching = false;
    private float pinchTimer = 0f;
    private GameObject previewMarkerInstance;
    private Vector3 currentHitPoint;

    /// <summary>
    /// We store the main "panel mesh" objects here.
    /// </summary>
    private List<GameObject> createdPanels = new List<GameObject>();

    private UnityUISlider holdSliderInstance;

    public float smoothSpeed = 10.0f; // Speed for smooth marker movement
    public float ballUpdateDistanceThreshhold = 2.0f; // Distance for raycasting

    /// <summary>
    /// Temporary corner markers (spawned on each pinch).
    /// Destroyed as soon as the panel is finalized OR if the user decides to cancel.
    /// </summary>
    private List<GameObject> temporaryCornerMarkers = new List<GameObject>();

    [Header("Hierarchy Settings")]
    public Transform panelParentTransform;


    void Awake()
    {
        corners = new Vector3[totalCorners];
        InitializePreviewMarker();
    }


    public void toogleMarkerVisibility()
    {
        if (previewMarkerInstance != null && panelMakingActive)
        {
            previewMarkerInstance.SetActive(!previewMarkerInstance.activeSelf);
        }
    }
    void InitializePreviewMarker()
    {
        if (previewMarkerPrefab == null) return;

        previewMarkerInstance = Instantiate(previewMarkerPrefab);
        previewMarkerInstance.layer = LayerMask.NameToLayer("Ignore Raycast");
        previewMarkerInstance.SetActive(panelMakingActive); // show/hide based on initial state

        var sliderCanvas = previewMarkerInstance.transform.Find("CircleSliderCanvas");
        if (sliderCanvas != null)
        {
            var sliderTransform = sliderCanvas.Find("Slider");
            if (sliderTransform != null)
            {
                holdSliderInstance = sliderTransform.GetComponent<UnityUISlider>();
                if (holdSliderInstance != null)
                {
                    holdSliderInstance.gameObject.SetActive(false);
                    holdSliderInstance.value = 0f;
                }
            }
        }
    }

    void Update()
    {
        // If panel-making is disabled, skip the corner-marking logic entirely
        if (!panelMakingActive)
            return;

        UpdateRaycast();
        UpdatePreviewMarker();
        UpdatePinchDetection();
    }

    void UpdateRaycast()
    {
        Ray camRay = new Ray(Camera.main.transform.position, Camera.main.transform.forward);
        int layerMask = ~(1 << LayerMask.NameToLayer("Ignore Raycast"));

        if (Physics.Raycast(camRay, out RaycastHit hit, Mathf.Infinity, layerMask))
        {
            currentHitPoint = hit.point;
        }
        else
        {
            currentHitPoint = camRay.origin + camRay.direction * 2f;
        }
    }

    void UpdatePreviewMarker()
    {
        if (previewMarkerInstance != null)
        {
            float distance = Vector3.Distance(previewMarkerInstance.transform.position, currentHitPoint);
            if (distance > ballUpdateDistanceThreshhold) // Adjust threshold as needed
            {
                previewMarkerInstance.transform.position = Vector3.Lerp(previewMarkerInstance.transform.position, currentHitPoint, Time.deltaTime * smoothSpeed);
            }
        }
    }

    void UpdatePinchDetection()
    {
        if (handController == null) return;

        float pinchAmount = handController.selectInteractionState.value;
        bool wasPinchingPreviously = isPinching;
        isPinching = pinchAmount > pinchThreshold;

        // Pinch started
        if (isPinching && !wasPinchingPreviously)
        {
            StartPinch();
        }
        // Pinch released
        else if (!isPinching && wasPinchingPreviously)
        {
            CancelPinch();
        }

        // Update pinch progress
        if (isPinching)
        {
            pinchTimer += Time.deltaTime;
            UpdateSlider(pinchTimer / pinchHoldTime);

            if (pinchTimer >= pinchHoldTime)
            {
                ConfirmCorner(currentHitPoint);
                CancelPinch();
            }
        }
    }

    void StartPinch()
    {
        pinchTimer = 0f;
        if (holdSliderInstance != null)
        {
            holdSliderInstance.gameObject.SetActive(true);
            holdSliderInstance.value = 0f;
        }
    }

    void CancelPinch()
    {
        isPinching = false;
        pinchTimer = 0f;
        if (holdSliderInstance != null)
        {
            holdSliderInstance.gameObject.SetActive(false);
            holdSliderInstance.value = 0f;
        }
    }

    void UpdateSlider(float progress)
    {
        if (holdSliderInstance != null)
        {
            holdSliderInstance.value = Mathf.Clamp01(progress);
        }
    }

    private void ConfirmCorner(Vector3 cornerPos)
    {
        if (currentCornerIndex >= totalCorners)
        {
            Debug.LogWarning("All corners already confirmed.");
            return;
        }

        corners[currentCornerIndex] = cornerPos;
        Debug.Log($"Corner {currentCornerIndex + 1} confirmed at {cornerPos}");

        // Create a temporary marker to show the user that the corner was placed
        if (confirmedCornerMarkerPrefab != null)
        {
            // Name them "TempCorner_#" so we know they're the placeholders
            GameObject tempMarker = Instantiate(confirmedCornerMarkerPrefab, cornerPos, Quaternion.identity);
            tempMarker.name = $"TempCorner_{currentCornerIndex + 1}";
            temporaryCornerMarkers.Add(tempMarker);
        }

        currentCornerIndex++;
        if (currentCornerIndex >= totalCorners)
        {
            BuildMeshAtCorners();
            ResetCornerPlacement();
        }
    }

    private void BuildMeshAtCorners()
    {
        if (corners.Length < 4)
        {
            Debug.LogError("Not enough corners to build the mesh.");
            return;
        }

        // --------------------------------------------------------------------
        // Create a parent GameObject in the desired hierarchy location
        // --------------------------------------------------------------------
        // If the user did not specify a parent in the Inspector,
        // we can default to this script's own transform or to the scene root.
        Transform targetParent = (panelParentTransform != null)
            ? panelParentTransform
            : null; // or "this.transform" if you want to fallback to the script's transform

        GameObject panelParent = new GameObject($"PanelParent_{createdPanels.Count + 1}");
        panelParent.transform.SetParent(targetParent, worldPositionStays: true);
        // --------------------------------------------------------------------

        // Create the panel
        GameObject panelObject = new GameObject("CustomPanelMesh");
        panelObject.transform.SetParent(panelParent.transform); // Attach to newly created parent

        MeshFilter mf = panelObject.AddComponent<MeshFilter>();
        MeshRenderer mr = panelObject.AddComponent<MeshRenderer>();

        Mesh mesh = new Mesh();
        Vector3[] vertices = new Vector3[] { corners[0], corners[1], corners[2], corners[3] };
        int[] triangles = new int[] { 0, 1, 2, 2, 3, 0 };
        Vector2[] uvs = new Vector2[]
        {
            new Vector2(0, 0),
            new Vector2(1, 0),
            new Vector2(1, 1),
            new Vector2(0, 1)
        };

        mesh.vertices = vertices;
        mesh.triangles = triangles;
        mesh.uv = uvs;
        mesh.RecalculateNormals();
        mf.mesh = mesh;

        // Assign the primary panel material
        if (panelMaterial != null)
        {
            mr.material = panelMaterial;
        }
        else
        {
            mr.material = new Material(Shader.Find("Standard"));
            mr.material.color = Color.gray;
        }

        // Keep track of this new panel
        createdPanels.Add(panelObject);

        AttachDeleteButton(panelObject);

        // Create final corner markers for the new panel
        CreateCornerMarkers(panelParent);

        // Destroy all temporary markers (so we don't see duplicates)
        DestroyTemporaryMarkers();

        Debug.Log("Panel mesh and markers created under parent GameObject.");
    }

    private void CreateCornerMarkers(GameObject parent)
    {
        for (int i = 0; i < corners.Length; i++)
        {
            if (confirmedCornerMarkerPrefab != null)
            {
                GameObject marker = Instantiate(confirmedCornerMarkerPrefab, corners[i], Quaternion.identity);
                marker.name = $"ConfirmedCorner_{i + 1}";
                marker.transform.SetParent(parent.transform); // Attach to parent
                Debug.Log($"Created final marker: {marker.name} at position {corners[i]}");
            }
        }
    }

    private void DestroyTemporaryMarkers()
    {
        // Destroys the placeholders used during corner confirmation
        foreach (var tempMarker in temporaryCornerMarkers)
        {
            if (tempMarker != null)
                Destroy(tempMarker);
        }
        temporaryCornerMarkers.Clear();
    }

    private void AttachDeleteButton(GameObject panelObject)
    {
        if (deleteButtonPrefab != null && panelObject != null)
        {
            // Instantiate the delete button as a child of the panel
            GameObject deleteButton = Instantiate(deleteButtonPrefab, panelObject.transform);

            // Calculate the center of the panel
            Vector3 panelCenter = Vector3.zero;
            foreach (Vector3 corner in corners)
            {
                panelCenter += corner;
            }
            panelCenter /= corners.Length; // Average position of all corners

            // Place the delete button at the calculated center
            deleteButton.transform.localPosition = panelCenter;
            deleteButton.transform.localRotation = Quaternion.identity;
            deleteButton.transform.localScale = Vector3.one;

            // Try to find the PressableButton component
            var pressableButton = deleteButton.GetComponent<PressableButton>();
            if (pressableButton != null)
            {
                // Dynamically add a listener to the button
                pressableButton.OnClicked.AddListener(() => DeleteWindow(panelObject));
                Debug.Log($"Delete button listener added for panel: {panelObject.name}");
            }
            else
            {
                Debug.LogWarning("Delete button prefab does not have a PressableButton component.");
            }
        }
        else
        {
            Debug.LogWarning("DeleteButtonPrefab or panelObject is null.");
        }
    }

    public void DeleteWindow(GameObject panel)
    {
        if (panel == null)
        {
            Debug.LogWarning("DeleteWindow: Panel is null!");
            return;
        }

        // -- IMPORTANT --
        // Remove from createdPanels so we don't keep a stale reference
        if (createdPanels.Contains(panel))
        {
            createdPanels.Remove(panel);
        }

        // Clean up the parent
        GameObject parent = panel.transform.parent?.gameObject;
        if (parent != null)
        {
            Destroy(parent);
            Debug.Log($"Deleted panel and its markers under parent: {parent.name}");
        }
        else
        {
            Debug.LogWarning("Panel has no parent, deleting panel only.");
            Destroy(panel);
        }
    }

    private void ResetCornerPlacement()
    {
        currentCornerIndex = 0;
        corners = new Vector3[totalCorners];
        Debug.Log("Corner placement reset for new window.");
    }

    // ---------------------------------------------------------------
    // NEW METHODS: CANCEL & UNDO
    // ---------------------------------------------------------------

    /// <summary>
    /// Cancel the in-progress panel:
    /// - Destroys all temporary corner markers
    /// - Resets the corner data so you start fresh
    /// </summary>
    public void CancelCurrentPanel()
    {
        // If no corners are placed, there's nothing to cancel
        if (currentCornerIndex == 0 && temporaryCornerMarkers.Count == 0)
        {
            Debug.Log("No panel in progress to cancel.");
            return;
        }

        // Destroy all temporary corner markers
        DestroyTemporaryMarkers();

        // Reset corner data
        ResetCornerPlacement();

        Debug.Log("Cancelled the in-progress panel creation.");
    }

    /// <summary>
    /// Undo the last corner placement:
    /// - Removes the last TempCorner marker (if any)
    /// - Adjusts the currentCornerIndex and clears that corner entry
    /// If there are no temporary corners, does nothing.
    /// </summary>
    public void UndoLastCorner()
    {
        if (currentCornerIndex <= 0 || temporaryCornerMarkers.Count == 0)
        {
            Debug.Log("No corner to undo.");
            return;
        }

        // Remove the last temporary marker
        int lastIndex = temporaryCornerMarkers.Count - 1;
        GameObject lastMarker = temporaryCornerMarkers[lastIndex];
        if (lastMarker != null)
        {
            Destroy(lastMarker);
        }
        temporaryCornerMarkers.RemoveAt(lastIndex);

        // Decrement currentCornerIndex
        currentCornerIndex--;

        // Clear the data for that corner
        corners[currentCornerIndex] = Vector3.zero;

        Debug.Log($"Undid corner {currentCornerIndex + 1} (now total corners = {currentCornerIndex}).");
    }

    // ---------------------------------------------------------------
    // METHODS FOR SWITCHING/REVERTING MATERIALS AND VISIBILITY
    // ---------------------------------------------------------------
    public void SwitchToAlternateMaterial()
    {
        if (alternatePanelMaterial == null)
        {
            Debug.LogWarning("No alternate panel material assigned.");
            return;
        }

        // First, remove any null entries (already destroyed) from our list
        createdPanels.RemoveAll(panel => panel == null);

        foreach (GameObject panelObject in createdPanels)
        {
            MeshRenderer mr = panelObject.GetComponent<MeshRenderer>();
            if (mr != null)
            {
                mr.material = alternatePanelMaterial;
            }

            Transform panelParent = panelObject.transform.parent;
            if (panelParent == null) continue;

            foreach (Transform child in panelParent.GetComponentsInChildren<Transform>(true))
            {
                // Hide final corner markers
                if (child.name.StartsWith("ConfirmedCorner_"))
                {
                    child.gameObject.SetActive(false);
                }

                // Hide the delete button
                var pressable = child.GetComponent<PressableButton>();
                if (pressable != null)
                {
                    child.gameObject.SetActive(false);
                }
            }
        }
    }

    public void RevertToOriginalMaterial()
    {
        // Remove any null entries
        createdPanels.RemoveAll(panel => panel == null);

        foreach (GameObject panelObject in createdPanels)
        {
            MeshRenderer mr = panelObject.GetComponent<MeshRenderer>();
            if (mr != null && panelMaterial != null)
            {
                mr.material = panelMaterial;
            }

            Transform panelParent = panelObject.transform.parent;
            if (panelParent == null) continue;

            foreach (Transform child in panelParent.GetComponentsInChildren<Transform>(true))
            {
                // Show final corner markers
                if (child.name.StartsWith("ConfirmedCorner_"))
                {
                    child.gameObject.SetActive(true);
                }

                // Show the delete button
                var pressable = child.GetComponent<PressableButton>();
                if (pressable != null)
                {
                    child.gameObject.SetActive(true);
                }
            }
        }
    }

    // ---------------------------------------------------------------
    // METHODS TO ENABLE / DISABLE THE PANEL-MAKING PROCESS
    // ---------------------------------------------------------------
    /// <summary>
    /// Disables the panel-making process:
    /// - No pinch detection
    /// - Hides the preview marker
    /// </summary>
    public void DisablePanelMaking()
    {
        panelMakingActive = false;
        if (previewMarkerInstance != null)
        {
            previewMarkerInstance.SetActive(false);
        }
    }

    /// <summary>
    /// Enables the panel-making process:
    /// - Reenables pinch detection
    /// - Shows the preview marker
    /// </summary>
    public void EnablePanelMaking()
    {
        panelMakingActive = true;
        if (previewMarkerInstance != null)
        {
            previewMarkerInstance.SetActive(true);
        }
    }
}
