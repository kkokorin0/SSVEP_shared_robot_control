using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using System.Threading;

public class StimServer : MonoBehaviour
{   // server
    public int connectionPort = 25001;
    TcpListener server;
    TcpClient client;
    Thread thread;
    bool running;

    // stimulus
    public float[] freqs = {0, 0, 0, 0}; //t, b, l, r
    public const int frameRate = 60;
    public const float dutyCycle = 0.5f;
    public int frameCount = 0;

    // movement
    public Vector3 gripper_pos = Vector3.zero;

    void Start()
    {   
        // receive on a separate thread so Unity doesn't freeze waiting for data
        ThreadStart ts = new ThreadStart(GetData);
        thread = new Thread(ts);
        thread.Start();
    }

    void GetData()
    {
        // create the server
        server = new TcpListener(IPAddress.Any, connectionPort);
        server.Start();

        // create a client to get the data stream
        client = server.AcceptTcpClient();

        // listen
        running = true;
        while (running)
        {
            Connection();
        }
        server.Stop();
    }

    void Connection()
    {
        // read data from the network stream
        NetworkStream nwStream = client.GetStream();
        byte[] buffer = new byte[client.ReceiveBufferSize];
        int bytesRead = nwStream.Read(buffer, 0, client.ReceiveBufferSize);

        // decode the bytes into a string
        string dataReceived = Encoding.UTF8.GetString(buffer, 0, bytesRead);

        // parse data is not empty
        if (dataReceived != null && dataReceived != "")
        {
            // process msg and update stim
            string[] msgArray = dataReceived.Split(':');
            if (msgArray[0] == "move")
            {
                UpdatePos(msgArray[1]);
            }
            else if (msgArray[0] == "reset")
            {
                gripper_pos = Vector3.zero; // reset pos
                frameCount = 0;             // reset frame count

                // update freqs
                for (int i = 0; i < freqs.Length; i++)
                {
                    if (float.TryParse(msgArray[1].Split(',')[i], out float floatValue))
                    {
                        freqs[i] = floatValue;
                    }
                    else
                    {
                        freqs[i] = 0; // failed parse
                    }
                }
            }

            // echo msg as response
            nwStream.Write(buffer, 0, bytesRead);
        }
    }

    // update gripper position
    void UpdatePos(string data)
    {
        string[] dataElements = data.Split(',');
        char direction = char.Parse(dataElements[0]);
        float stepSize = float.Parse(dataElements[1]);

        // move right, left, up, down, forward or back by step m
        if (direction == 'r')
        {
            gripper_pos += new Vector3(stepSize, 0, 0);
        }
        else if (direction == 'l')
        {
            gripper_pos += new Vector3(-stepSize, 0, 0);
        }
        else if (direction == 'u')
        {
            gripper_pos += new Vector3(0, stepSize, 0);
        }
        else if (direction == 'd')
        {
            gripper_pos += new Vector3(0, -stepSize, 0);
        }
        else if (direction == 'f')
        {
            gripper_pos += new Vector3(0, 0, -stepSize);
        }
        else if (direction == 'b')
        {
            gripper_pos += new Vector3(0, 0, stepSize);
        }
    }

    void Update()
    {
        // Could update stim position more frequently then messages received
        frameCount += 1;
    }
}