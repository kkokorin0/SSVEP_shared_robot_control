using UnityEngine;
using Unity.Collections;
using Unity.Networking.Transport;

public class ServerBehaviour : MonoBehaviour
{

    NetworkDriver m_Driver;
    NativeList<NetworkConnection> m_Connections;

    void Start()
    {
    }

    void OnDestroy()
    {
    }

    void Update()
    {
    }
}