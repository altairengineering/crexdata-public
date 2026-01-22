using UnityEngine;

public class VShapeArea : MonoBehaviour
{
    public float angle = 45f; // Angle of the V shape in degrees
    public float length = 10f; // Length of each arm of the V shape
    public Color gizmoColor = Color.red; // Color of the V shape in the editor
    public Color fillColor = new Color(1f, 0f, 0f, 0.5f); // Semi-transparent fill color
    public bool isEnabled = true; // To enable/disable the V shape

    private Transform vShapeTransform;
    private Mesh vShapeMesh;
    private GameObject vShapeObject;
    private GameObject lineObject;
    private LineRenderer lineRenderer;
    private MeshRenderer meshRenderer;
    private MeshFilter meshFilter;

    void CreateVShape()
    {
        DestroyVShape(); // Ensure no duplicates

        vShapeObject = new GameObject("VShape");
        vShapeTransform = vShapeObject.transform;
        vShapeTransform.SetParent(transform);
        vShapeTransform.localPosition = new Vector3(0, -2, 6); // 2 units below the camera
        vShapeTransform.localRotation = Quaternion.identity;

        meshFilter = vShapeObject.AddComponent<MeshFilter>();
        meshRenderer = vShapeObject.AddComponent<MeshRenderer>();
        meshRenderer.material = new Material(Shader.Find("Standard"));
        meshRenderer.material.color = fillColor;

        CreateVShapeMesh();
    }

    void CreateVShapeMesh()
    {
        vShapeMesh = new Mesh();

        Vector3[] vertices = new Vector3[3];
        int[] triangles = new int[3];
        Vector3[] normals = new Vector3[3];

        vertices[0] = Vector3.zero;
        vertices[1] = Quaternion.Euler(0, -angle / 2, 0) * Vector3.forward * length;
        vertices[2] = Quaternion.Euler(0, angle / 2, 0) * Vector3.forward * length;

        triangles[0] = 0;
        triangles[1] = 1;
        triangles[2] = 2;

        // Compute normals
        normals[0] = Vector3.up;
        normals[1] = Vector3.up;
        normals[2] = Vector3.up;

        vShapeMesh.vertices = vertices;
        vShapeMesh.triangles = triangles;
        vShapeMesh.normals = normals;

        meshFilter.mesh = vShapeMesh;
    }

    void DestroyVShape()
    {
        if (vShapeObject != null)
        {
            DestroyImmediate(vShapeObject);
        }
    }

    void CreateLineRenderer()
    {
        DestroyLineRenderer(); // Ensure no duplicates

        lineObject = new GameObject("VShapeLines");
        lineObject.transform.SetParent(transform);
        lineObject.transform.localPosition = new Vector3(0, -2, 6); // 2 units below the camera
        lineObject.transform.localRotation = Quaternion.identity;

        lineRenderer = lineObject.AddComponent<LineRenderer>();
        lineRenderer.startWidth = 0.02f;
        lineRenderer.endWidth = 0.02f;
        lineRenderer.positionCount = 3;
        lineRenderer.material = new Material(Shader.Find("Sprites/Default"));
        lineRenderer.startColor = gizmoColor;
        lineRenderer.endColor = gizmoColor;
        lineRenderer.useWorldSpace = false;

        UpdateLineRenderer();
    }

    void DestroyLineRenderer()
    {
        if (lineObject != null)
        {
            DestroyImmediate(lineObject);
        }
    }

    void Update()
    {
        // Enable or disable the V shape
        if (vShapeObject) vShapeObject.SetActive(isEnabled);
        if (lineObject) lineObject.SetActive(isEnabled);

        // Update the V shape if enabled
        if (isEnabled)
        {
            UpdateVShape();
            UpdateLineRenderer();
        }

        // Position both 2 units below the main camera
        //if (vShapeObject) vShapeObject.transform.position = Camera.main.transform.position + Vector3.down * 2;
       // if (lineObject) lineObject.transform.position = Camera.main.transform.position + Vector3.down * 2;
    }

    void UpdateVShape()
    {
        if (vShapeMesh == null) return;

        Vector3[] vertices = new Vector3[3];
        vertices[0] = Vector3.zero;
        vertices[1] = Quaternion.Euler(0, -angle / 2, 0) * Vector3.forward * length;
        vertices[2] = Quaternion.Euler(0, angle / 2, 0) * Vector3.forward * length;

        vShapeMesh.vertices = vertices;
        vShapeMesh.RecalculateNormals();
    }

    void UpdateLineRenderer()
    {
        if (lineRenderer == null) return;

        Vector3[] positions = new Vector3[3];
        positions[0] = Vector3.zero;
        positions[1] = Quaternion.Euler(0, -angle / 2, 0) * Vector3.forward * length;
        positions[2] = Quaternion.Euler(0, angle / 2, 0) * Vector3.forward * length;
        lineRenderer.SetPositions(positions);
    }

    private void Start()
    {
        DestroyVShape();
        DestroyLineRenderer();
        CreateVShape();
        CreateLineRenderer();
    }
}
