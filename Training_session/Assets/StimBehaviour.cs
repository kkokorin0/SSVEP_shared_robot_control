using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class StimBehaviour : MonoBehaviour
{
    public StimServer server;
    public Vector3 offset;

    // init stim
    void Start()
    {
        transform.position = server.gripper_pos + offset;
    }

    // update stim pos
    void Update()
    {
        transform.position = server.gripper_pos + offset;
    }
}
