using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AlignManager : MonoBehaviour
{

    public GameObject alignObject;
    //public GameObject originObject;

    //main camera object
    public Camera mainCamera;

    public int deegrees;

    public void Align()
    {
        //Destroy the alignObject from my scene
        //Destroy(alignObject);
        //Calculate the angle between the given deegrees and the main camera
        float angle = alignObject.transform.rotation.eulerAngles.y - mainCamera.transform.rotation.eulerAngles.y;
        Debug.Log("Angle: " + angle);
        //Change the align objects Y rotation to the calculated angle
        alignObject.transform.rotation = Quaternion.Euler(0, angle, 0);
        this.gameObject.SetActive(false);


    }
}
