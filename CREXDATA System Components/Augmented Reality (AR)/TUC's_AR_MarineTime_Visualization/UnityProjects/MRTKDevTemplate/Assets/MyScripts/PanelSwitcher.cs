using UnityEngine;

public class PanelSwitcher : MonoBehaviour
{
    // Reference to the panel2 prefab
    public GameObject panel2Prefab;

    public void SwitchPanel()
    {
        // Capture the size of panel1 accurately at the moment of switching
        Vector3 panel1Size = GetSize(gameObject);
        if (panel1Size == Vector3.zero)
        {
            Debug.LogError("Failed to get panel3 size.");
            return;
        }

        // Debug log to ensure the correct size is captured
        Debug.Log("Panel1 Size: " + panel1Size);

        // Instantiate panel2 at the same position and rotation as panel1
        GameObject panel2 = Instantiate(panel2Prefab, transform.position, transform.rotation);

        // Find the specific child with the renderer named "UX.Slate.MagicWindow"
        Transform magicWindowTransform = panel2.transform.Find("UX.Slate.MagicWindow");
        if (magicWindowTransform != null)
        {
            GameObject magicWindow = magicWindowTransform.gameObject;

            // Adjust the size of the "UX.Slate.MagicWindow" child to match the size of panel1
            SetSize(magicWindow, panel1Size);

            // Debug log to confirm size adjustment
            Debug.Log("MagicWindow Size After Instantiation: " + GetSize(magicWindow));
        }
        else
        {
            Debug.LogError("Child 'UX.Slate.MagicWindow' not found in panel2 prefab.");
        }

        // Destroy panel1 after panel2 has been instantiated
        Destroy(gameObject);
    }

    private Vector3 GetSize(GameObject obj)
    {
        Renderer renderer = obj.GetComponent<Renderer>();
        if (renderer != null)
        {
            return renderer.bounds.size;
        }

        RectTransform rectTransform = obj.GetComponent<RectTransform>();
        if (rectTransform != null)
        {
            return new Vector3(rectTransform.rect.width, rectTransform.rect.height, rectTransform.rect.width);
        }

        Debug.LogError("Renderer or RectTransform not found on object: " + obj.name);
        return Vector3.zero;
    }

    private void SetSize(GameObject obj, Vector3 size)
    {
        Renderer renderer = obj.GetComponent<Renderer>();
        if (renderer != null)
        {
            Vector3 originalSize = renderer.bounds.size;

            // Avoid division by zero
            if (originalSize.x == 0 || originalSize.y == 0 || originalSize.z == 0)
            {
                Debug.LogError("Original size of the object is zero, cannot scale properly.");
                return;
            }

            Vector3 scale = obj.transform.localScale;
            Vector3 newScale = new Vector3(
                scale.x * (size.x / originalSize.x),
                scale.y * (size.y / originalSize.y),
                scale.z * (size.z / originalSize.z)
            );

            obj.transform.localScale = newScale;
        }
        else
        {
            RectTransform rectTransform = obj.GetComponent<RectTransform>();
            if (rectTransform != null)
            {
                rectTransform.sizeDelta = new Vector2(size.x, size.y);
            }
            else
            {
                Debug.LogError("Renderer or RectTransform not found on object: " + obj.name);
            }
        }
    }
}
