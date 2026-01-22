using Mapbox.Unity.MeshGeneration.Factories;
using System.Collections;
using UnityEngine;

public class Billboard : MonoBehaviour
{
    public Transform cameraTransform;
    public float rotationSpeed = 5f; // Speed of rotation towards the camera

    private Quaternion targetRotation;

    private Transform parentTranform;

    public bool enableForParent = false;
    private bool isSmoothing = false;


    void Start()
    {
        //If it has a parent
        if (transform.parent != null)
            parentTranform = transform.parent;
    }

    void LateUpdate()
    {
        if (cameraTransform == null)
        {
            cameraTransform = Camera.main.transform;
        }

        if (!isSmoothing)
        {
            // Update the object to face the camera, ignoring pitch
            Vector3 targetPosition = new Vector3(cameraTransform.position.x, transform.position.y, cameraTransform.position.z);
            targetRotation = Quaternion.LookRotation(-targetPosition + transform.position);
            transform.rotation = targetRotation;
            //Rotate parent too
            if (enableForParent && parentTranform != null)
                parentTranform.rotation = targetRotation;
        }
    }

    public void SmoothLookAtCamera()
    {
        if (!isSmoothing)
        {
            StartCoroutine(SmoothLookAt());
        }
    }

    private IEnumerator SmoothLookAt()
    {
        isSmoothing = true;
        Quaternion initialRotation = transform.rotation;
        Vector3 targetPosition = new Vector3(cameraTransform.position.x, transform.position.y, cameraTransform.position.z);
        targetRotation = Quaternion.LookRotation(-targetPosition + transform.position);

        float elapsedTime = 0f;
        while (elapsedTime < 0.5f)
        {
            transform.rotation = Quaternion.Slerp(initialRotation, targetRotation, elapsedTime * rotationSpeed);
            if (enableForParent && parentTranform != null)
                parentTranform.rotation = transform.rotation; // Rotate parent too
            elapsedTime += Time.deltaTime;
            yield return null;
        }

        transform.rotation = targetRotation;
        if (enableForParent && parentTranform != null)
            parentTranform.rotation = transform.rotation; // Rotate parent too
        isSmoothing = false;
    }
}
