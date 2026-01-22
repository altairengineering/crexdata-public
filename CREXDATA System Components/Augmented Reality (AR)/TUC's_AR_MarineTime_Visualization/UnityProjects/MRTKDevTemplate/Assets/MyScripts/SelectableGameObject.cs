using UnityEngine;

[System.Serializable]
public class SelectableGameObject
{
    public GameObject gameObject;
    public bool selected;

    public SelectableGameObject(GameObject obj, bool isSelected)
    {
        gameObject = obj;
        selected = isSelected;
    }
}
