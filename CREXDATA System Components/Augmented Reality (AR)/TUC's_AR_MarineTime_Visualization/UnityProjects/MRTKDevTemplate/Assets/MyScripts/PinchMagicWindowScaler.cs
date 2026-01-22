//using Microsoft.MixedReality.Toolkit;
using UnityEngine;

public class PinchMagicWindowScaler : MonoBehaviour
{
    public GameObject magicWindowPrefab; // Reference to the MagicWindow prefab
    private GameObject magicWindowInstance;

    private Vector3 firstCorner;
    private Vector3 secondCorner;
    private bool firstCornerSelected = false;

    void OnEnable()
    {
        PinchEventHandler.OnPinchStart += HandlePinchStart;
        PinchEventHandler.OnPinchEnd += HandlePinchEnd;
    }

    void OnDisable()
    {
        PinchEventHandler.OnPinchStart -= HandlePinchStart;
        PinchEventHandler.OnPinchEnd -= HandlePinchEnd;
    }

    private void HandlePinchStart(Vector3 position)
    {
        if (!firstCornerSelected)
        {
            firstCorner = position;
            firstCornerSelected = true;
            CreateNewMagicWindow();
        }
    }

    private void HandlePinchEnd(Vector3 position)
    {
        if (firstCornerSelected)
        {
            secondCorner = position;
            AdjustMagicWindow();
        }
    }

    private void CreateNewMagicWindow()
    {
        if (magicWindowInstance != null)
        {
            Destroy(magicWindowInstance);
        }

        magicWindowInstance = Instantiate(magicWindowPrefab);
        magicWindowInstance.SetActive(true);
        magicWindowInstance.transform.position = firstCorner;
        magicWindowInstance.transform.localScale = Vector3.one;
    }

    private void AdjustMagicWindow()
    {
        if (magicWindowInstance != null)
        {
            Vector3 center = (firstCorner + secondCorner) / 2;
            Vector3 size = new Vector3(Mathf.Abs(firstCorner.x - secondCorner.x), Mathf.Abs(firstCorner.y - secondCorner.y), 0.01f);

            // Adjust the MagicWindow's position and scale
            magicWindowInstance.transform.position = center;
            magicWindowInstance.transform.localScale = size;
        }

        firstCornerSelected = false; // Reset for new area selection
    }
}
