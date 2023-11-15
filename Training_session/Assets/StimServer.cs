using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using System.Threading;

public class StimServer : MonoBehaviour
{   // server
    private int _connectionPort = 25001;
    TcpListener _server;
    TcpClient _client;
    Thread _thread;
    bool _running;

    // stimulus
    public float StartTime;
    public float[] Freqs = {0, 0, 0, 0}; //t, b, l, r

    // movement
    private Vector3 _gripperStart = new(0, 0, 2);
    public Vector3 GripperPos;

    void Start()
    {
        StartTime = Time.time;
        GripperPos = _gripperStart;

        // setup the world coordinates

        // receive on a separate thread so Unity doesn't freeze waiting for data
        ThreadStart ts = new(GetData);
        _thread = new Thread(ts);
        _thread.Start();
    }

    void GetData()
    {
        // create the server
        _server = new TcpListener(IPAddress.Any, _connectionPort);
        _server.Start();

        // create a client to get the data stream
        _client = _server.AcceptTcpClient();

        // listen
        _running = true;
        while (_running)
        {
            Connection();
        }
        _server.Stop();
    }

    void Connection()
    {
        // read data from the network stream
        var nwStream = _client.GetStream();
        var buffer = new byte[_client.ReceiveBufferSize];
        var bytesRead = nwStream.Read(buffer, 0, _client.ReceiveBufferSize);

        // decode the bytes into a string
        var dataReceived = Encoding.UTF8.GetString(buffer, 0, bytesRead);

        // data is empty
        if (dataReceived == null || dataReceived == "") return;
        
        // process msg and update stim
        var msgArray = dataReceived.Split(':');
        Debug.Log(msgArray[0]);
        Debug.Log(msgArray[1]);
        
        switch (msgArray[0])
        {
            case "move":
                UpdatePos(msgArray[1]);
                break;

            case "reset":
                // update freqs
                for (var i = 0; i < Freqs.Length; i++)
                {
                    Freqs[i] = 0; // set to 0 if can't parse
                    if (float.TryParse(msgArray[1].Split(',')[i], out float floatValue))
                    {
                        Freqs[i] = floatValue;
                    }
                }

                GripperPos = _gripperStart;  // reset pos
                break;
        }

        // echo msg as response
        nwStream.Write(buffer, 0, bytesRead);
        
    }

    // update gripper position
    void UpdatePos(string data)
    {
        var dataElements = data.Split(',');
        var direction = char.Parse(dataElements[0]);
        var stepSize = float.Parse(dataElements[1]);

        // move right, left, up, down, forward or back by step m
        switch (direction)
        {
            case 'r':
            GripperPos += new Vector3(stepSize, 0, 0);
            break;

            case 'l':
            GripperPos += new Vector3(-stepSize, 0, 0);
            break;

            case 'u':
            GripperPos += new Vector3(0, stepSize, 0);
            break;

            case 'd':
            GripperPos += new Vector3(0, -stepSize, 0);
            break;

            case 'f':
            GripperPos += new Vector3(0, 0, -stepSize);
            break;

            case 'b':
            GripperPos += new Vector3(0, 0, stepSize);
            break;
        }
    }

    void Update()
    {
    }
}