using UnityEngine;

public class StimBehaviour : MonoBehaviour
{
    public StimServer Server;       // comms with python
    public Vector3 Offset;          // gripper offset
    public int BlockIndex;          // t, b, l, r
    const float MaxSize = 0.08f;    // m
    
    void Start()
    // init stim
    {
        transform.localPosition = Server.GripperPos + Offset;
        transform.localScale = new Vector3(MaxSize, MaxSize, MaxSize);
    }


    void Update()
    // update stim pos and flash
    {
        transform.position = Server.GripperPos + Offset;

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
            //Debug.Log(string.Format("Freq: {0}, time: {1}, A: {2}, size: {3}", 
            //    server.freqs[blockIndex], elapsedTime, amplitude, blockSize))
        }
    }
}
