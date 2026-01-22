using UnityEngine;
using System.Collections.Generic;
using UnityEngine.InputSystem;  // For InputAction
using UnityEngine.UI;          // For Slider
using MixedReality.Toolkit.UX; // If you use MRTK's UX components

// To avoid ambiguity between UnityEngine.UI.Slider and MixedReality.Toolkit.UX.Slider, 
// alias the Unity UI Slider class:
using UnityUISlider = UnityEngine.UI.Slider;

public class MultiWindowCornerMarkerWithDeleteButton : MonoBehaviour
{
    [Header("Corner Settings")]
    public int totalCorners = 4;

    [Header("Prefabs (Optional)")]
    public GameObject previewMarkerPrefab;
    public GameObject confirmedCornerMarkerPrefab;

    [Header("Hold Gesture Settings")]
    [Tooltip("Time (in seconds) the user must hold left-click to confirm a corner.")]
    public float leftClickHoldTime = 2.0f;

    [Header("Mesh Material")]
    public Material panelMaterial;

    [Header("Delete Button")]
    [Tooltip("Prefab of the delete button to place on each panel.")]
    public GameObject deleteButtonPrefab;

    [Header("Mouse Click Input Action")]
    [Tooltip("Reference to the InputActionAsset that has a 'LeftClickHold' action.")]
    public InputActionAsset inputActionsAsset;

    [Tooltip("Name of the Action Map containing the LeftClickHold action.")]
    public string actionMapName = "MouseActions";       // Example
    [Tooltip("Name of the action in the Action Map (e.g., 'LeftClickHold').")]
    public string leftClickActionName = "LeftClickHold"; // Example

    private InputAction leftClickHoldAction;

    private Vector3[] corners;
    private int currentCornerIndex = 0;

    private bool isLeftClickHeld = false;
    private float leftClickTimer = 0f;

    private GameObject previewMarkerInstance;
    private Vector3 currentHitPoint;

    private List<GameObject> createdPanels = new List<GameObject>();

    // Reference to the Unity UI Slider in the marker prefab
    private UnityUISlider holdSliderInstance;

    void Awake()
    {
        corners = new Vector3[totalCorners];

        // 1) Instantiate the marker prefab
        if (previewMarkerPrefab != null)
        {
           // previewMarkerInstance = Instantiate(previewMarkerPrefab, Vector3.zero, Quaternion.identity);

            // Assign to Ignore Raycast layer to avoid self-raycast hits
            int ignoreLayer = LayerMask.NameToLayer("Ignore Raycast");
            previewMarkerInstance.layer = ignoreLayer;

            previewMarkerInstance.SetActive(true);

            // 2) Find the Slider in the prefab (if any)
            var sliderCanvas = previewMarkerInstance.transform.Find("CircleSliderCanvas");
            if (sliderCanvas != null)
            {
                var sliderTransform = sliderCanvas.Find("Slider");
                if (sliderTransform != null)
                {
                    holdSliderInstance = sliderTransform.GetComponent<UnityUISlider>();
                    if (holdSliderInstance != null)
                    {
                        // Hide the slider initially
                        holdSliderInstance.gameObject.SetActive(false);
                        holdSliderInstance.value = 0f;
                    }
                }
            }
        }

        // 3) Load the left-click hold action from the InputActionAsset
        if (inputActionsAsset != null)
        {
            var actionMap = inputActionsAsset.FindActionMap(actionMapName);
            if (actionMap != null)
            {
                leftClickHoldAction = actionMap.FindAction(leftClickActionName);
                if (leftClickHoldAction == null)
                {
                    Debug.LogWarning($"LeftClickHold action '{leftClickActionName}' not found in '{actionMapName}' Action Map.");
                }
            }
            else
            {
                Debug.LogWarning($"Action Map '{actionMapName}' not found in the InputActionAsset.");
            }
        }
        else
        {
            Debug.LogWarning("InputActionsAsset is not assigned.");
        }
    }

    void OnEnable()
    {
        // 4) Enable the action and subscribe to events
        if (leftClickHoldAction != null)
        {
            leftClickHoldAction.Enable();
            leftClickHoldAction.started += OnLeftClickStarted;
            leftClickHoldAction.canceled += OnLeftClickCanceled;
        }
    }

    void OnDisable()
    {
        // Unsubscribe and disable the action
        if (leftClickHoldAction != null)
        {
            leftClickHoldAction.started -= OnLeftClickStarted;
            leftClickHoldAction.canceled -= OnLeftClickCanceled;
            leftClickHoldAction.Disable();
        }
    }

    // When user presses left-click
    private void OnLeftClickStarted(InputAction.CallbackContext ctx)
    {
        isLeftClickHeld = true;
        leftClickTimer = 0f;

        if (holdSliderInstance != null)
        {
            holdSliderInstance.gameObject.SetActive(true);
            holdSliderInstance.value = 0f;
        }
    }

    // When user releases left-click
    private void OnLeftClickCanceled(InputAction.CallbackContext ctx)
    {
        if (isLeftClickHeld)
        {
            ResetLeftClickHold();
        }
    }

    void Update()
    {
        // Keep the marker at the raycast hit or a default distance
        Ray camRay = new Ray(Camera.main.transform.position, Camera.main.transform.forward);
        int ignoreLayer = LayerMask.NameToLayer("Ignore Raycast");
        int layerMask = ~(1 << ignoreLayer);

        if (Physics.Raycast(camRay, out RaycastHit hit, Mathf.Infinity, layerMask))
        {
            currentHitPoint = hit.point;
        }
        else
        {
            currentHitPoint = camRay.origin + camRay.direction * 2f;
        }

        if (previewMarkerInstance != null)
        {
            previewMarkerInstance.transform.position = currentHitPoint;
        }

        // Update hold timer & slider if user is holding left-click
        if (isLeftClickHeld && holdSliderInstance != null)
        {
            leftClickTimer += Time.deltaTime;
            float fillValue = Mathf.Clamp01(leftClickTimer / leftClickHoldTime);
            holdSliderInstance.value = fillValue;

            // Check if hold duration completed
            if (leftClickTimer >= leftClickHoldTime)
            {
                ConfirmCorner(currentHitPoint);
                ResetLeftClickHold();
            }
        }
    }

    private void ResetLeftClickHold()
    {
        isLeftClickHeld = false;
        leftClickTimer = 0f;
        if (holdSliderInstance != null)
        {
            holdSliderInstance.gameObject.SetActive(false);
            holdSliderInstance.value = 0f;
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

        if (confirmedCornerMarkerPrefab != null)
        {
            GameObject marker = Instantiate(confirmedCornerMarkerPrefab, cornerPos, Quaternion.identity);
            marker.name = $"ConfirmedCorner_{currentCornerIndex + 1}";
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

        DeleteExistingMarkers();

        // Create a parent for the new panel
        GameObject panelParent = new GameObject($"PanelParent_{createdPanels.Count + 1}");

        // Create the mesh
        GameObject panelObject = new GameObject("CustomPanelMesh");
        panelObject.transform.SetParent(panelParent.transform);

        MeshFilter mf = panelObject.AddComponent<MeshFilter>();
        MeshRenderer mr = panelObject.AddComponent<MeshRenderer>();

        Mesh mesh = new Mesh();
        Vector3[] vertices = { corners[0], corners[1], corners[2], corners[3] };
        int[] triangles = { 0, 1, 2, 2, 3, 0 };
        Vector2[] uvs =
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

        if (panelMaterial != null)
        {
            mr.material = panelMaterial;
        }
        else
        {
            mr.material = new Material(Shader.Find("Standard"));
            mr.material.color = Color.gray;
        }

        createdPanels.Add(panelObject);
        AttachDeleteButton(panelObject);

        CreateCornerMarkers(panelParent);

        Debug.Log("Panel mesh and markers created under parent GameObject.");
    }

    private void DeleteExistingMarkers()
    {
        for (int i = 0; i < totalCorners; i++)
        {
            string markerName = $"ConfirmedCorner_{i + 1}";
            GameObject existing = GameObject.Find(markerName);
            if (existing != null)
            {
                Destroy(existing);
            }
        }
    }

    private void CreateCornerMarkers(GameObject parent)
    {
        for (int i = 0; i < corners.Length; i++)
        {
            if (confirmedCornerMarkerPrefab != null)
            {
                GameObject marker = Instantiate(confirmedCornerMarkerPrefab, corners[i], Quaternion.identity);
                marker.name = $"ConfirmedCorner_{i + 1}";
                marker.transform.SetParent(parent.transform);
            }
        }
    }

    private void AttachDeleteButton(GameObject panelObject)
    {
        if (deleteButtonPrefab != null && panelObject != null)
        {
            GameObject deleteButton = Instantiate(deleteButtonPrefab, panelObject.transform);

            // Compute the centroid
            Vector3 center = Vector3.zero;
            foreach (var c in corners)
            {
                center += c;
            }
            center /= corners.Length;

            deleteButton.transform.localPosition = center;
            deleteButton.transform.localRotation = Quaternion.identity;
            deleteButton.transform.localScale = Vector3.one;

            var pressableButton = deleteButton.GetComponent<PressableButton>();
            if (pressableButton != null)
            {
                pressableButton.OnClicked.AddListener(() => DeleteWindow(panelObject));
            }
        }
    }

    public void DeleteWindow(GameObject panel)
    {
        if (panel == null) return;
        var parent = panel.transform.parent;
        if (parent != null)
        {
            Destroy(parent.gameObject);
        }
        else
        {
            Destroy(panel);
        }
    }

    private void ResetCornerPlacement()
    {
        currentCornerIndex = 0;
        corners = new Vector3[totalCorners];
    }
}
