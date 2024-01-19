using UnityEngine;

public class StimBehaviour : MonoBehaviour
{
    public StimServer Server;       // comms with python
    public Vector3 Offset;          // gripper offset
    public int BlockIndex;          // t, b, l, r.m
    const float MaxSize = 0.052f;    // m
    
    void Start()
    // init stim
    {
        transform.localPosition = Server.GripperPose.position + Offset;
        transform.localScale = new Vector3(MaxSize, MaxSize, MaxSize);
    }


    void Update()
    // update stim pos and flash
    {
        transform.position = Server.GripperPose.position + Server.GripperPose.rotation * Offset;
        transform.rotation = Server.GripperPose.rotation;

        if (Server.Freqs[BlockIndex] == 0)
        {
            // don't flash
            transform.localScale = new Vector3(MaxSize, MaxSize, MaxSize);
        }
        else
        {
            // create square wave for block size
            var elapsedTime = Time.time - Server.StartTime;
            var amplitude = Mathf.Sin(2 * Mathf.PI * Server.Freqs[BlockIndex] * elapsedTime);
            var blockSize = MaxSize / 2 * (Mathf.Sign(amplitude) + 1);

            transform.localScale = new Vector3(blockSize, blockSize, blockSize);
        }
    }
}
