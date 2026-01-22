using UnityEngine;
using System.Collections.Generic;
using MixedReality.Toolkit.SpatialManipulation;
using MixedReality.Toolkit;

// Extended SelectableGameObject to also store the panel instance
[System.Serializable]
public class CustomSelectableGameObject
{
    public GameObject gameObject;
    public GameObject panelInstance; // Stores the panel for each object

    public CustomSelectableGameObject(GameObject gameObject, GameObject panelInstance = null)
    {
        this.gameObject = gameObject;
        this.panelInstance = panelInstance;
    }
}

public class GenerateObjectInFront : MonoBehaviour
{
    [Header("Prefabs")]
    public GameObject magicWindow;   // The prefab for the MagicWindow
    public GameObject panelPrefab;   // The prefab for the Panel

    // Callback to notify about new objects
    public System.Action<GameObject> OnObjectGenerated;

    // List to store generated objects (each with its own panel)
    public List<CustomSelectableGameObject> generatedObjects = new List<CustomSelectableGameObject>();

    /// <summary>
    /// Generates a new MagicWindow prefab 1 unit in front of the main camera,
    /// and also creates a unique panel that is linked to that MagicWindow.
    /// </summary>
    public void GenerateObject()
    {
        if (magicWindow == null)
        {
            Debug.LogError("MagicWindow prefab is not assigned.");
            return;
        }

        // Find camera and compute spawn position
        Transform cameraTransform = Camera.main.transform;
        Vector3 spawnPosition = cameraTransform.position + cameraTransform.forward * 1.0f;

        // Instantiate the MagicWindow
        GameObject newObject = Instantiate(magicWindow, spawnPosition, cameraTransform.rotation);
        newObject.name = $"MagicWindow_{generatedObjects.Count + 1}";

        // Instantiate a Panel as a separate object
        GameObject newPanel = null;
        if (panelPrefab != null)
        {
            Vector3 panelPosition = spawnPosition + (Vector3.up * 0.5f); // Offset above the object
            Quaternion panelRotation = Quaternion.identity;
            newPanel = Instantiate(panelPrefab, panelPosition, panelRotation);
            newPanel.name = $"Panel_for_{newObject.name}";
        }
        else
        {
            Debug.LogWarning("PanelPrefab is not assigned. No panel will be created for this object.");
        }

        // Create a new CustomSelectableGameObject that stores both the MagicWindow and its Panel
        CustomSelectableGameObject selectableObject = new CustomSelectableGameObject(newObject, newPanel);
        generatedObjects.Add(selectableObject);

        // Add MaintainConstantGapUsingBounds to ensure proper spacing between MagicWindow and Panel
        if (newPanel != null)
        {
            MaintainConstantGapUsingBounds gapScript = newObject.AddComponent<MaintainConstantGapUsingBounds>();
            gapScript.primaryWindow = newObject;   // Set MagicWindow as primary
            gapScript.secondaryWindow = newPanel;  // Set Panel as secondary
            gapScript.fixedGap = 0.1f;             // Set the default gap
        }

        // Make sure the object has StatefulInteractable
        if (newObject.GetComponent<StatefulInteractable>() == null)
        {
            newObject.AddComponent<StatefulInteractable>();
        }

        // No more “SelfSelector” usage (we removed it).
        // If you still want to do something on click, you can do it here:
        // StatefulInteractable interactable = newObject.GetComponent<StatefulInteractable>();
        // interactable.OnClicked.AddListener(() => Debug.Log($"Clicked on {newObject.name}"));

        // Notify any listeners about the new object
        OnObjectGenerated?.Invoke(newObject);
    }

    /// <summary>
    /// Disables manipulation and selection for all generated objects,
    /// and also disables the panel associated with each object.
    /// </summary>
    public void DisableManipulationAndSelection()
    {
        foreach (CustomSelectableGameObject selectable in generatedObjects)
        {
            // Disable ObjectManipulator if it exists
            ObjectManipulator manipulator = selectable.gameObject.GetComponent<ObjectManipulator>();
            if (manipulator != null)
            {
                manipulator.enabled = false;
            }

            // Disable StatefulInteractable to prevent selection
            StatefulInteractable interactable = selectable.gameObject.GetComponent<StatefulInteractable>();
            if (interactable != null)
            {
                interactable.enabled = false;
            }

            // Disable the associated panel
            if (selectable.panelInstance != null)
            {
                selectable.panelInstance.SetActive(false);
            }
        }
    }

    /// <summary>
    /// Enables manipulation and selection for all generated objects,
    /// and also enables the panel associated with each object.
    /// </summary>
    public void EnableManipulationAndSelection()
    {
        foreach (CustomSelectableGameObject selectable in generatedObjects)
        {
            // Enable ObjectManipulator if it exists
            ObjectManipulator manipulator = selectable.gameObject.GetComponent<ObjectManipulator>();
            if (manipulator != null)
            {
                manipulator.enabled = true;
            }

            // Enable StatefulInteractable to allow selection
            StatefulInteractable interactable = selectable.gameObject.GetComponent<StatefulInteractable>();
            if (interactable != null)
            {
                interactable.enabled = true;
            }

            // Enable the associated panel
            if (selectable.panelInstance != null)
            {
                selectable.panelInstance.SetActive(true);
            }
        }
    }

    /// <summary>
    /// Disables all panels, disables the "UX.Slate.ContentBackplate" of each MagicWindow,
    /// and ensures all MagicWindows are frozen.
    /// </summary>
    [Header("Scripts")]
    public ScalingScript scalingScript; // Reference to ScalingScript
    public List<GameObject> objectsFrozenByDisableAllPanels = new List<GameObject>();

    public void DisableAllPanels()
    {
        objectsFrozenByDisableAllPanels.Clear(); // Clear the list before starting

        foreach (CustomSelectableGameObject selectable in generatedObjects)
        {
            // Disable the associated panel
            if (selectable.panelInstance != null)
            {
                selectable.panelInstance.SetActive(false);
            }

            // Find and disable the "UX.Slate.ContentBackplate" child of the MagicWindow
            if (selectable.gameObject != null)
            {
                Transform magicWindowTransform = selectable.gameObject.transform.Find("UX.Slate.MagicWindow");
                if (magicWindowTransform != null)
                {
                    Transform contentBackplate = magicWindowTransform.Find("UX.Slate.ContentBackplate");
                    if (contentBackplate != null)
                    {
                        contentBackplate.gameObject.SetActive(false);
                    }
                }

                // Ensure the MagicWindow is frozen
                if (scalingScript != null && scalingScript.cachedTransforms.TryGetValue(selectable.gameObject, out var transforms))
                {
                    if (!scalingScript.objectFrozenStates.ContainsKey(selectable.gameObject) 
                        || !scalingScript.objectFrozenStates[selectable.gameObject])
                    {
                        scalingScript.ToggleFreezeForObject(selectable.gameObject); // Freeze the MagicWindow
                        objectsFrozenByDisableAllPanels.Add(selectable.gameObject); // Track the object
                    }
                }
            }
        }
    }

    /// <summary>
    /// Enables all panels and the "UX.Slate.ContentBackplate" of each MagicWindow.
    /// </summary>
    public void EnableAllPanels()
    {
        foreach (CustomSelectableGameObject selectable in generatedObjects)
        {
            // Enable the associated panel
            if (selectable.panelInstance != null)
            {
                selectable.panelInstance.SetActive(true);
            }

            // Find and enable the "UX.Slate.ContentBackplate" child of the MagicWindow
            if (selectable.gameObject != null)
            {
                Transform magicWindowTransform = selectable.gameObject.transform.Find("UX.Slate.MagicWindow");
                if (magicWindowTransform != null)
                {
                    Transform contentBackplate = magicWindowTransform.Find("UX.Slate.ContentBackplate");
                    if (contentBackplate != null)
                    {
                        contentBackplate.gameObject.SetActive(true);
                    }
                }
            }
        }

        // Unfreeze only the objects that were frozen during DisableAllPanels
        foreach (GameObject obj in objectsFrozenByDisableAllPanels)
        {
            if (scalingScript != null)
            {
                scalingScript.ToggleFreezeForObject(obj); // Unfreeze the MagicWindow
            }
        }

        // Clear the list after unfreezing
        objectsFrozenByDisableAllPanels.Clear();
    }
}
