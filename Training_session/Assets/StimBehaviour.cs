using System.Collections;
using System.Collections.Generic;
using Unity.Networking.Transport.TLS;
using UnityEngine;
using UnityEngine.Timeline;

public class StimBehaviour : MonoBehaviour
{
    public StimServer server;       // comms with python
    public Vector3 offset;          // gripper offset
    public int blockIndex;          // t, b, l, r
    const float blockSize = 0.08f;  // m
    int toggleFrame = 0;            // last frame block changed

    // init stim
    void Start()
    {
        transform.localPosition = server.gripper_pos + offset;
        transform.localScale = new Vector3(blockSize, blockSize, blockSize);
    }

    // update stim pos and flash
    void Update()
    {
        transform.position = server.gripper_pos + offset;

        int onFrames = StimServer.frameRate;
        if (server.freqs[blockIndex] != 0) // do nothing if freq == 0
        {
            onFrames = (int)Mathf.Round(StimServer.frameRate / server.freqs[blockIndex] *
                StimServer.dutyCycle);

            if (server.frameCount > toggleFrame + onFrames)
            {
                if (transform.localScale == Vector3.zero) // turn on
                {
                    transform.localScale = new Vector3(blockSize, blockSize, blockSize);
                }
                else
                {
                    transform.localScale = Vector3.zero; // turn off
                    toggleFrame = server.frameCount;
                }
            }
        }
        //Debug.Log(server.freqs[blockIndex]);
        //Debug.Log(onFrames);

    }
}
