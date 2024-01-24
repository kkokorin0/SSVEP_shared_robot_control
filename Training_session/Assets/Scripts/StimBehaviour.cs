using UnityEngine;

public class StimBehaviour : MonoBehaviour
{
    /* Flashing SSVEP stimulus with frequency (Hz) and position controlled by StimServer.cs.
       The stimulus is designed to appear relative to the position of a robot gripper and
       be used to control the directional motion of the gripper. */

    public StimServer Server;       // stimulus controller
    public Vector3 Offset;          // stimulus offset relative to gripper
    public int BlockIndex;          // which direction the stimulus represents
    const float MaxSize = 0.052f;   // stimulus size
    private Renderer _stim_render;  // stimulus material renderer
    
    void Start()
    // Setup stimulus relative to gripper position
    {
        transform.localPosition = Server.GripperPose.position + Offset;
        transform.localScale = new Vector3(MaxSize, MaxSize, MaxSize);
        _stim_render = gameObject.GetComponent<Renderer>();
    }


    void Update()
    // Update stimulus position and transparency based on StimServer parameters
    {
        // locate stimulus relative to gripper
        transform.position = Server.GripperPose.position + Server.GripperPose.rotation * Offset;
        transform.rotation = Server.GripperPose.rotation;
        var stimColor = _stim_render.material.color;

        switch (Server.Freqs[BlockIndex])
        {
            case 0:
                // turn off
                stimColor.a = 0;
                break;

            case 1:
                // turn on without flashing
                stimColor.a = 1;
                break;

            default:
                // square wave flashing (50% duty cycle) by varying stimulus size
                var elapsedTime = Time.time - Server.StartTime;
                var amplitude = Mathf.Sin(2 * Mathf.PI * Server.Freqs[BlockIndex] * elapsedTime);
                stimColor.a = 0.5f * (Mathf.Sign(amplitude) + 1);
                break;
        }
        _stim_render.material.color = stimColor;
    }
}
