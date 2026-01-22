using System.Collections.Generic;
using UnityEngine;

public class MapTogglesInfoPossitioning : MonoBehaviour
{
    [SerializeField] private List<GameObject> objectsToShow = new List<GameObject>();

    // Preset relative positions for each object relative to the camera's local space.
    [SerializeField] private List<Vector3> relativePositions = new List<Vector3>
    {
        new Vector3(0, 0, 2),
        new Vector3(1, 0, 2),
        new Vector3(-1, 0, 2),
        new Vector3(0, 1, 2)
    };

    /// <summary>
    /// Call this method from your UI button to show and position objects.
    /// </summary>
    public void OnShowObjectsButtonPressed() {
        ShowObjectsFacingUser(relativePositions);
    }

    /// <summary>
    /// Positions each object relative to the camera, activates them, makes them face the user,
    /// and applies any special rotations needed for specific objects.
    /// </summary>
    /// <param name="relPositions">List of relative positions for each object.</param>
    public void ShowObjectsFacingUser(List<Vector3> relPositions)
    {
        Camera mainCamera = Camera.main;
        if (mainCamera == null) {
            Debug.LogError("Main camera not found.");
            return;
        }

        if (objectsToShow.Count != relPositions.Count) {
            Debug.LogError("Mismatch between number of objects and relative positions.");
            return;
        }

        for (int i = 0; i < objectsToShow.Count; i++)
        {
            GameObject obj = objectsToShow[i];
            Vector3 relPos = relPositions[i];

            // Calculate world position relative to the camera.
            Vector3 spawnPosition = mainCamera.transform.TransformPoint(relPos);
            obj.transform.position = spawnPosition;

            if (!obj.activeSelf) {
                obj.SetActive(true);
            }

            // Make the object face the camera while staying upright.
            Vector3 lookTarget = mainCamera.transform.position;
            lookTarget.y = spawnPosition.y;
            obj.transform.LookAt(lookTarget);

            // Rotate 180° around the Y-axis to ensure proper facing.
            obj.transform.Rotate(0, 180f, 0);

            // Apply additional rotation for a specific object if needed.
            // Change "SpecificObjectName" to the name of the object that needs extra rotation.
            if (obj.name == "Map") {
                // Rotate -90 degrees around the X-axis.
                obj.transform.Rotate(-90f, 0f, 0f);
            }
        }
    }
}
