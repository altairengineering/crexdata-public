using System.Collections.Generic;
using UnityEngine;
using MixedReality.Toolkit.SpatialManipulation;
using MixedReality.Toolkit.UX;

public class ScalingScript : MonoBehaviour
{
    private List<CustomSelectableGameObject> generatedObjects; 
    public Dictionary<GameObject, (Transform magicWindow, Transform backPlate)> cachedTransforms;

    // We keep per-object scaling states, so each MagicWindow can be manipulated independently
    private Dictionary<GameObject, ScalingState> objectScalingStates = new Dictionary<GameObject, ScalingState>();
    // If you want to freeze each object independently, store that state here
    public Dictionary<GameObject, bool> objectFrozenStates = new Dictionary<GameObject, bool>();

    public string objectSpawnerName = "ObjectSpawner";           // Name of the ObjectSpawner GameObject in the hierarchy
    public string magicWindowName   = "UX.Slate.MagicWindow";    // Name of the child GameObject to act on

    // Speeds/scales
    public float tiltSpeed      = 4f;    // Speed for tilting
    public float rotationSpeed  = 4f;    // Speed for rotation
    public float scaleSpeed     = 0.1f;  // Speed for scaling

    // One for each MagicWindow’s initial scale
    private Dictionary<GameObject, Vector3> magicWindowOriginalScales = new Dictionary<GameObject, Vector3>();

    // One for each BackPlate’s initial scale
    private Dictionary<GameObject, Vector3> backPlateOriginalScales   = new Dictionary<GameObject, Vector3>();

    // **Define the list to track frozen objects**
    private List<GameObject> objectsFrozenByDisableAllPanels = new List<GameObject>();

    public enum ScalingState
    {
        None,
        ScalingUpX,
        ScalingUpY,
        ScalingDownX,
        ScalingDownY,
        TiltingUp,
        TiltingDown,
        RotatingClockwise,
        RotatingCounterClockwise
    }

    void Start()
    {
        // 1. Find the GenerateObjectInFront script and subscribe to new-object creation
        GameObject objectSpawner = GameObject.Find(objectSpawnerName);
        if (objectSpawner != null)
        {
            GenerateObjectInFront generateObjectInFront = objectSpawner.GetComponent<GenerateObjectInFront>();
            if (generateObjectInFront != null)
            {
                generatedObjects = generateObjectInFront.generatedObjects;
                generateObjectInFront.OnObjectGenerated += AddToCache; // Subscribe
            }
            else
            {
                Debug.LogError($"GenerateObjectInFront component not found on {objectSpawnerName}.");
            }
        }
        else
        {
            Debug.LogError($"ObjectSpawner named '{objectSpawnerName}' not found in the scene.");
        }

        // 2. Cache transforms for already existing objects
        CacheTransforms();

        // 3. Hook up each existing object’s panel to this script
        HookUpExistingPanels();
    }

    /// <summary>
    /// For every object in generatedObjects, we store references to its MagicWindow and BackPlate,
    /// and also set up default states in the dictionaries.
    /// </summary>
    private void CacheTransforms()
    {
        cachedTransforms = new Dictionary<GameObject, (Transform magicWindow, Transform backPlate)>();

        foreach (var item in generatedObjects)
        {
            if (item.gameObject == null) 
                continue;

            var magicWindow = item.gameObject.transform.Find(magicWindowName);
            var backPlate   = magicWindow?.Find("BackPlate");

            if (magicWindow == null)
            {
                Debug.LogWarning($"MagicWindow '{magicWindowName}' not found in '{item.gameObject.name}'.");
                continue;
            }

            // Remember the transforms
            cachedTransforms[item.gameObject] = (magicWindow, backPlate);

            // Default states
            if (!objectScalingStates.ContainsKey(item.gameObject))
                objectScalingStates[item.gameObject] = ScalingState.None;

            if (!objectFrozenStates.ContainsKey(item.gameObject))
                objectFrozenStates[item.gameObject] = false;

            // Store per-object initial scales (if valid)
            if (magicWindow != null && backPlate != null)
            {
                // Save the initial localScales for this pair
                magicWindowOriginalScales[item.gameObject] = magicWindow.localScale;
                backPlateOriginalScales[item.gameObject]   = backPlate.localScale;
            }
        }
    }

    private void AddToCache(GameObject newObject)
    {
        var magicWindow = newObject.transform.Find(magicWindowName);
        var backPlate   = magicWindow?.Find("BackPlate");

        if (magicWindow != null && backPlate != null)
        {
            cachedTransforms[newObject] = (magicWindow, backPlate);

            // Store their original scales
            magicWindowOriginalScales[newObject] = magicWindow.localScale;
            backPlateOriginalScales[newObject]   = backPlate.localScale;
        }
        else
        {
            Debug.LogError($"Failed to find '{magicWindowName}' or 'BackPlate' in '{newObject.name}'.");
            return;
        }

        // Initialize default states
        if (!objectScalingStates.ContainsKey(newObject))
            objectScalingStates[newObject] = ScalingState.None;

        if (!objectFrozenStates.ContainsKey(newObject))
            objectFrozenStates[newObject] = false;

        // Hook up the panel’s buttons, etc.
        var cso = generatedObjects.Find(x => x.gameObject == newObject);
        if (cso != null && cso.panelInstance != null)
        {
            HookUpPanelButtons(newObject, cso.panelInstance);
        }
    }

    /// <summary>
    /// Because some objects might have been pre-spawned in the scene,
    /// we also try hooking up each object’s panel references (if any).
    /// </summary>
    private void HookUpExistingPanels()
    {
        foreach (var item in generatedObjects)
        {
            if (item.panelInstance != null)
            {
                HookUpPanelButtons(item.gameObject, item.panelInstance);
            }
        }
    }

    /// <summary>
    /// Looks up the PressableButtons in the panel and wires them up to methods that act on 'thisObject'.
    /// </summary>
    private void HookUpPanelButtons(GameObject thisObject, GameObject panel)
    {
        // "PushToSet"
        var pushToSetBtn = panel.transform.Find("PushToSet")?.GetComponent<PressableButton>();
        if (pushToSetBtn != null)
            pushToSetBtn.OnClicked.AddListener(() => ToggleFreezeForObject(thisObject));

        // "Scale Up Y"
        var scaleUpYBtn = panel.transform.Find("Scale Up Y")?.GetComponent<PressableButton>();
        if (scaleUpYBtn != null)
            scaleUpYBtn.OnClicked.AddListener(() => SetScalingStateForObject(thisObject, ScalingState.ScalingUpY));

        // "Scale Down Y"
        var scaleDownYBtn = panel.transform.Find("Scale Down Y")?.GetComponent<PressableButton>();
        if (scaleDownYBtn != null)
            scaleDownYBtn.OnClicked.AddListener(() => SetScalingStateForObject(thisObject, ScalingState.ScalingDownY));

        // "Tilt Down"
        var tiltDownBtn = panel.transform.Find("Tilt Down")?.GetComponent<PressableButton>();
        if (tiltDownBtn != null)
            tiltDownBtn.OnClicked.AddListener(() => SetScalingStateForObject(thisObject, ScalingState.TiltingDown));

        // "Tilt Up"
        var tiltUpBtn = panel.transform.Find("Tilt Up")?.GetComponent<PressableButton>();
        if (tiltUpBtn != null)
            tiltUpBtn.OnClicked.AddListener(() => SetScalingStateForObject(thisObject, ScalingState.TiltingUp));

        // "Scale Up X"
        var scaleUpXBtn = panel.transform.Find("Scale Up X")?.GetComponent<PressableButton>();
        if (scaleUpXBtn != null)
            scaleUpXBtn.OnClicked.AddListener(() => SetScalingStateForObject(thisObject, ScalingState.ScalingUpX));

        // "Scale Down X"
        var scaleDownXBtn = panel.transform.Find("Scale Down X")?.GetComponent<PressableButton>();
        if (scaleDownXBtn != null)
            scaleDownXBtn.OnClicked.AddListener(() => SetScalingStateForObject(thisObject, ScalingState.ScalingDownX));

        // "Rotate Counterclockwise" (fixed typo)
        var rotateCCWBtn = panel.transform.Find("Rotate Counterclockwise")?.GetComponent<PressableButton>();
        if (rotateCCWBtn != null)
            rotateCCWBtn.OnClicked.AddListener(() => SetScalingStateForObject(thisObject, ScalingState.RotatingCounterClockwise));

        // "Rotate Clockwise"
        var rotateCWBtn = panel.transform.Find("Rotate Clockwise")?.GetComponent<PressableButton>();
        if (rotateCWBtn != null)
            rotateCWBtn.OnClicked.AddListener(() => SetScalingStateForObject(thisObject, ScalingState.RotatingClockwise));

        // **New: "Delete" Button**
        var deleteBtn = panel.transform.Find("Delete")?.GetComponent<PressableButton>();
        if (deleteBtn != null)
            deleteBtn.OnClicked.AddListener(() => DeleteObject(thisObject));
        else
            Debug.LogWarning($"Delete button not found in panel '{panel.name}'. Ensure the button is named 'Delete' and has a PressableButton component.");
    }

    /// <summary>
    /// Sets (or toggles off) the scaling state for a particular object. 
    /// If you click the same button again, it reverts to None.
    /// </summary>
    private void SetScalingStateForObject(GameObject obj, ScalingState state)
    {
        if (!objectScalingStates.ContainsKey(obj)) return;
        // If user clicks the same button twice, toggle it off
        objectScalingStates[obj] = (objectScalingStates[obj] == state) ? ScalingState.None : state;
    }

    /// <summary>
    /// Toggles "frozen" state for this particular object, disabling object manipulations if frozen,
    /// and deactivating the ObjectManipulator of the parent object of UX.Slate.MagicWindow.
    /// </summary>
    public void ToggleFreezeForObject(GameObject obj)
    {
        if (!objectFrozenStates.ContainsKey(obj))
            objectFrozenStates[obj] = false;

        objectFrozenStates[obj] = !objectFrozenStates[obj];
        bool isFrozen = objectFrozenStates[obj];

        // If we previously cached transforms:
        if (cachedTransforms.TryGetValue(obj, out var transforms))
        {
            var (magicWindow, backPlate) = transforms;

            if (magicWindow != null)
            {
                // Deactivate the ObjectManipulator of the parent of UX.Slate.MagicWindow
                var parentObject = magicWindow.parent;
                if (parentObject != null)
                {
                    var objectManipulator = parentObject.GetComponent<ObjectManipulator>();
                    if (objectManipulator != null)
                    {
                        objectManipulator.enabled = !isFrozen;
                    }
                }
            }

            // Optionally hide/show the backPlate
            if (backPlate != null)
            {
                backPlate.gameObject.SetActive(!isFrozen);
            }
        }

        Debug.Log($"{obj.name} is now " + (isFrozen ? "frozen." : "unfrozen."));
    }

    /// <summary>
    /// Deletes a MagicWindow and its associated Panel, removing them from all lists and dictionaries.
    /// </summary>
    public void DeleteObject(GameObject obj)
    {
        if (obj == null)
        {
            Debug.LogError("DeleteObject called with a null GameObject.");
            return;
        }

        // Find the CustomSelectableGameObject where obj is either the magicWindow or the panel.
        CustomSelectableGameObject cso = generatedObjects.Find(x => x.gameObject == obj || x.panelInstance == obj);

        if (cso != null)
        {
            // Remove from generatedObjects list
            generatedObjects.Remove(cso);

            // Remove from cachedTransforms
            if (cachedTransforms.ContainsKey(cso.gameObject))
                cachedTransforms.Remove(cso.gameObject);

            // Remove from scaling and frozen states
            if (objectScalingStates.ContainsKey(cso.gameObject))
                objectScalingStates.Remove(cso.gameObject);

            if (objectFrozenStates.ContainsKey(cso.gameObject))
                objectFrozenStates.Remove(cso.gameObject);

            if (magicWindowOriginalScales.ContainsKey(cso.gameObject))
                magicWindowOriginalScales.Remove(cso.gameObject);

            if (backPlateOriginalScales.ContainsKey(cso.gameObject))
                backPlateOriginalScales.Remove(cso.gameObject);

            // Remove from objectsFrozenByDisableAllPanels if present
            if (objectsFrozenByDisableAllPanels.Contains(cso.gameObject))
                objectsFrozenByDisableAllPanels.Remove(cso.gameObject);

            // Destroy the GameObjects
            if (cso.gameObject != null)
            {
                Destroy(cso.gameObject);
                Debug.Log($"Deleted MagicWindow '{cso.gameObject.name}'.");
            }

            if (cso.panelInstance != null)
            {
                Destroy(cso.panelInstance);
                Debug.Log($"Deleted Panel '{cso.panelInstance.name}'.");
            }
        }
        else
        {
            Debug.LogError($"DeleteObject: Could not find the object to delete for '{obj.name}'.");
        }
    }

    /// <summary>
    /// Main update loop: we iterate all objects and see if they have a non-None state + are not frozen.
    /// Then we apply the corresponding scale/tilt/rotate operation.
    /// </summary>
    void Update()
    {
        if (generatedObjects == null) return;

        foreach (var item in generatedObjects)
        {
            GameObject obj = item.gameObject;
            if (obj == null) 
                continue;

            // If this object is frozen, skip
            bool isFrozen = objectFrozenStates.ContainsKey(obj) && objectFrozenStates[obj];
            if (isFrozen) 
                continue;

            // If we have a scaling state for the object, act on it
            if (objectScalingStates.TryGetValue(obj, out ScalingState state))
            {
                switch (state)
                {
                    case ScalingState.ScalingUpX:
                        AdjustScale(obj, new Vector3(scaleSpeed * Time.deltaTime, 0, 0));
                        break;
                    case ScalingState.ScalingUpY:
                        AdjustScale(obj, new Vector3(0, scaleSpeed * Time.deltaTime, 0));
                        break;
                    case ScalingState.ScalingDownX:
                        AdjustScale(obj, new Vector3(-scaleSpeed * Time.deltaTime, 0, 0));
                        break;
                    case ScalingState.ScalingDownY:
                        AdjustScale(obj, new Vector3(0, -scaleSpeed * Time.deltaTime, 0));
                        break;
                    case ScalingState.TiltingUp:
                        AdjustTilt(obj, Vector3.right * tiltSpeed * Time.deltaTime);  // tilt up
                        break;
                    case ScalingState.TiltingDown:
                        AdjustTilt(obj, Vector3.left * tiltSpeed * Time.deltaTime);   // tilt down
                        break;
                    case ScalingState.RotatingClockwise:
                        AdjustRotation(obj, Vector3.up * rotationSpeed * Time.deltaTime); // rotate cw
                        break;
                    case ScalingState.RotatingCounterClockwise:
                        AdjustRotation(obj, Vector3.down * rotationSpeed * Time.deltaTime);// rotate ccw
                        break;
                }
            }
        }
    }

    private void AdjustScale(GameObject obj, Vector3 scaleChange)
    {
        if (!cachedTransforms.TryGetValue(obj, out var transforms))
            return;

        var (magicWindow, backPlate) = transforms;
        if (magicWindow == null || backPlate == null)
            return;

        // 1) Scale the MagicWindow
        magicWindow.localScale += scaleChange;
        Vector3 mwScale = magicWindow.localScale;

        // Optionally, enforce minimum and maximum scales
        // Example:
        // magicWindow.localScale = Vector3.Max(magicWindow.localScale, Vector3.one * 0.5f);
        // magicWindow.localScale = Vector3.Min(magicWindow.localScale, Vector3.one * 2f);
    }

    private void AdjustTilt(GameObject obj, Vector3 tiltChange)
    {
        if (!cachedTransforms.TryGetValue(obj, out var transforms))
            return;

        var (magicWindow, backPlate) = transforms;
        if (magicWindow == null) return;

        // Tilt the MagicWindow in its own local space
        magicWindow.Rotate(tiltChange, Space.Self);

        // Match backPlate’s rotation, preserve scale
        if (backPlate != null)
        {
            backPlate.rotation = magicWindow.rotation;
        }
    }

    private void AdjustRotation(GameObject obj, Vector3 rotationChange)
    {
        if (!cachedTransforms.TryGetValue(obj, out var transforms))
            return;

        var (magicWindow, backPlate) = transforms;
        if (magicWindow == null) return;

        // Rotate in world space (if you prefer local, change to Space.Self)
        magicWindow.Rotate(rotationChange, Space.World);
    }
}
