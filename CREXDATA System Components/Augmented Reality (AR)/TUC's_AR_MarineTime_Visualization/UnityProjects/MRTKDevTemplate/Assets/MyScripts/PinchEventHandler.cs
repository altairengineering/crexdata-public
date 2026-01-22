using MixedReality.Toolkit.Subsystems;
using MixedReality.Toolkit;
using UnityEngine;

public class PinchEventHandler : MonoBehaviour
{
    public delegate void PinchEventHandlerDelegate(Vector3 position);
    public static event PinchEventHandlerDelegate OnPinchStart;
    public static event PinchEventHandlerDelegate OnPinchEnd;

    private HandsAggregatorSubsystem handSubsystem;

    private void Start()
    {
        handSubsystem = XRSubsystemHelpers.GetFirstRunningSubsystem<HandsAggregatorSubsystem>();
    }

    private void Update()
    {
        if (handSubsystem != null)
        {
            foreach (Handedness handedness in System.Enum.GetValues(typeof(Handedness)))
            {
                if (handedness == Handedness.None) continue;

                if (handSubsystem.TryGetPinchProgress(handedness, out bool isReadyToPinch, out bool isPinching, out float pinchAmount))
                {
                    if (isPinching)
                    {
                        //if (handSubsystem.TryGetJointPose(TrackedHandJoint.IndexTip, handedness, out HandJointPose pose))
                        {
                            //OnPinchStart?.Invoke(pose.Position);
                        }
                    }
                    else
                    {
                        //if (handSubsystem.TryGetJointPose(TrackedHandJoint.IndexTip, handedness, out HandJointPose pose))
                        {
                            //OnPinchEnd?.Invoke(pose.Position);
                        }
                    }
                }
            }
        }
    }
}
