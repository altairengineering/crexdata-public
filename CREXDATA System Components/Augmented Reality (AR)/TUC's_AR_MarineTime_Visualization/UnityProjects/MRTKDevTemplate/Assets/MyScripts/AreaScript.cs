using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AreaScript : MonoBehaviour
{
    private MeshRenderer renderer;

    private void Start()
    {
        // Get the MeshRenderer component
        renderer = GetComponent<MeshRenderer>();

        // Check if the MeshRenderer component is missing
        if (renderer == null)
        {
            Debug.LogError("MeshRenderer component is missing on " + gameObject.name);
        }
    }
    void OnDrawGizmos()
    {
        if (GetComponent<MeshCollider>())
        {
            Gizmos.color = Color.yellow;
            Gizmos.DrawWireMesh(GetComponent<MeshCollider>().sharedMesh, transform.position, transform.rotation, transform.localScale);
        }
    }

    void OnTriggerEnter(Collider other)
    {
        Debug.Log(other.gameObject.tag);
        // Check if the other collider is null
        if (other == null)
        {
            Debug.LogError("Other collider is null");
            return;
        }

        // Check if the other collider is the player
        if (other.gameObject.tag == "Player")
        {
            // Check if the MeshRenderer component is missing
            if (renderer == null)
            {
                return;
            }

            // Change the color of the material
            renderer.material.color = Color.red;

            // Print a message to the console
            Debug.Log("Player entered the cube area");
        }
    }

    void OnTriggerExit(Collider other)
    {
        // Check if the other collider is null
        if (other == null)
        {
            Debug.LogError("Other collider is null");
            return;
        }

        // Check if the other collider is the player
        if (other.gameObject.tag == "Player")
        {
            // Check if the MeshRenderer component is missing
            if (renderer == null)
            {
                return;
            }

            // Change the color of the material back to its original state
            renderer.material.color = Color.white;

            // Print a message to the console
            Debug.Log("Player exited the cube area");
        }
    }
}
