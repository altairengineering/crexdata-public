using System.Collections;
using UnityEngine;
using UnityEngine.XR;
using MixedReality.Toolkit.Subsystems;
using MixedReality.Toolkit;

public class PinchConnector : MonoBehaviour
{
    public GameObject squarePrefab; // Assign the square prefab in the inspector

    private HandsAggregatorSubsystem aggregator;
    private bool firstPinchConfirmed = false;
    private Vector3 firstPinchPosition;
    private float lastPinchTime = 0f;
    private int pinchCount = 0;
    private const float pinchInterval = 1.0f; // 1-second interval for detecting double pinch
    private const float minWaitTime = 1.0f; // Minimum wait time before accepting the second double pinch
    private float lastDoublePinchTime = 0f;

    void Start()
    {
        StartCoroutine(EnableWhenSubsystemAvailable());
    }

    IEnumerator EnableWhenSubsystemAvailable()
    {
        yield return new WaitUntil(() => XRSubsystemHelpers.GetFirstRunningSubsystem<HandsAggregatorSubsystem>() != null);
        aggregator = XRSubsystemHelpers.GetFirstRunningSubsystem<HandsAggregatorSubsystem>();
        Debug.Log("HandsAggregatorSubsystem enabled.");
    }

    void Update()
    {
        if (aggregator != null)
        {
            bool isPinching;
            Vector3 pinchPosition;

            if (IsPinching(XRNode.LeftHand, out isPinching, out pinchPosition) || IsPinching(XRNode.RightHand, out isPinching, out pinchPosition))
            {
                if (isPinching)
                {
                    float currentTime = Time.time;

                    if (currentTime - lastPinchTime <= pinchInterval)
                    {
                        pinchCount++;
                    }
                    else
                    {
                        pinchCount = 1;
                    }

                    lastPinchTime = currentTime;

                    if (pinchCount == 2) // Detect double pinch
                    {
                        if (currentTime - lastDoublePinchTime > minWaitTime)
                        {
                            Debug.Log("Double pinch detected.");
                            ProcessPinch(pinchPosition);
                            lastDoublePinchTime = currentTime;
                        }
                    }
                }
            }
        }
    }

    bool IsPinching(XRNode hand, out bool isPinching, out Vector3 pinchPosition)
    {
        isPinching = false;
        pinchPosition = Vector3.zero;

        if (aggregator.TryGetPinchProgress(hand, out bool isReadyToPinch, out bool isCurrentlyPinching, out float pinchAmount))
        {
            if (isCurrentlyPinching)
            {
                if (aggregator.TryGetJoint(TrackedHandJoint.IndexTip, hand, out HandJointPose jointPose))
                {
                    isPinching = true;
                    pinchPosition = jointPose.Position;
                    Debug.Log($"Pinch detected at position: {pinchPosition}");
                    return true;
                }
            }
        }

        return false;
    }

    void ProcessPinch(Vector3 pinchPosition)
    {
        if (!firstPinchConfirmed)
        {
            firstPinchConfirmed = true;
            firstPinchPosition = pinchPosition;
            Debug.Log($"First pinch confirmed at position: {firstPinchPosition}");
        }
        else
        {
            CreateSquare(firstPinchPosition, pinchPosition);
            firstPinchConfirmed = false;
        }
    }

    void CreateSquare(Vector3 firstPosition, Vector3 secondPosition)
    {
        Vector3 centerPosition = (firstPosition + secondPosition) / 2;
        Vector3 scale = new Vector3(Mathf.Abs(firstPosition.x - secondPosition.x), Mathf.Abs(firstPosition.y - secondPosition.y), Mathf.Abs(firstPosition.z - secondPosition.z));

        GameObject newSquare = Instantiate(squarePrefab, centerPosition, Quaternion.identity);
        newSquare.transform.localScale = scale;
        Debug.Log($"Square created between {firstPosition} and {secondPosition} with center at {centerPosition} and scale {scale}");
    }
}
