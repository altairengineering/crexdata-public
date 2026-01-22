using UnityEngine;

public class ConstantScreenSize : MonoBehaviour
{
    public Camera cameraToUse;
    public float objectSize = 1.0f;  // Desired object size

    void Update()
    {
        if (cameraToUse == null)
            cameraToUse = Camera.main; // Use main camera if none specified

        float distance = Vector3.Distance(transform.position, cameraToUse.transform.position);
        float objectScale = distance * objectSize / (cameraToUse.fieldOfView * Mathf.Tan(Mathf.Deg2Rad * 0.5f));
        transform.localScale = Vector3.one * objectScale;
    }
}
