using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using System.Threading;
using Microsoft.MixedReality.OpenXR;
using Microsoft.MixedReality.SampleQRCodes;

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
    private Vector3 _gripperStart = new(0.199f, -0.311f, -0.283f);
    private Vector3 _robotToQR = new(-0.182f, 0.378f, 0.414f);
    private Vector3 _gripperPos;
    public Pose GripperPose;

    // QR code tracking
    public QRCodesManager QrCodesManager;
    private Microsoft.MixedReality.QR.QRCode _qrCode;
    private Guid _qrCoords;
    private bool _qrFound = false;
    public Pose QRCodePose;
    public GameObject QRCodeFrame;
    public GameObject DummyQR;

    void Start()
    {
        StartTime = Time.time;
        _gripperPos = _gripperStart;
        GripperPose = new Pose(_gripperPos, Quaternion.identity);

        // setup QR detection
        QrCodesManager.StartQRTracking();

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
                UpdateGripperPose(QRCodePose, _robotToQR, _gripperPos);
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

                // reset pos
                _gripperPos = _gripperStart;  
                UpdateGripperPose(QRCodePose, _robotToQR, _gripperPos);
                break;
        }

        // echo msg as response
        nwStream.Write(buffer, 0, bytesRead);
        
    }

    void UpdatePos(string data)
    // update gripper position
    {
        var dataElements = data.Split(',');
        _gripperPos = new Vector3(float.Parse(dataElements[0]), float.Parse(dataElements[1]),
            float.Parse(dataElements[2]));
    }
    
    void UpdateGripperPose(Pose QRPose, Vector3 RobotOrigin, Vector3 GripperPos)
    {
        var gripperPos = QRPose.position + QRPose.rotation * (RobotOrigin + GripperPos);
        GripperPose = new Pose(gripperPos, QRPose.rotation);
    }
    
    void Update()
    {
        // look for QR code and setup global coords
        if (QrCodesManager.GetList().Count == 0)
        {
            // use simulated QR code
            QRCodePose = new Pose(DummyQR.transform.position, DummyQR.transform.rotation);
            UpdateGripperPose(QRCodePose, _robotToQR, _gripperPos);
            QRCodeFrame.transform.SetPositionAndRotation(GripperPose.position, GripperPose.rotation);
            Debug.Log("Using simulated QR code");
        }
        else
        {
            // actual QR code found
            _qrCode = QrCodesManager.GetList()[0];
            _qrCoords = _qrCode.SpatialGraphNodeId;

            SpatialGraphNode coordinateSystem = SpatialGraphNode.FromStaticNodeId(_qrCoords);
            coordinateSystem.TryLocate(FrameTime.OnUpdate, out QRCodePose);
            UpdateGripperPose(QRCodePose, _robotToQR, _gripperPos);
            QRCodeFrame.transform.SetPositionAndRotation(GripperPose.position, GripperPose.rotation);

            // destroy helper objects
            /*QrCodesManager.StopQRTracking();
            Destroy(QrCodesManager);*/
            Destroy(DummyQR);
            Destroy(QRCodeFrame);
            _qrFound = true;
        }
    }
}