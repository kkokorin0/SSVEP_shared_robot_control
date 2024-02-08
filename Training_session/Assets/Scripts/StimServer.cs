using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using System.Threading;
using Microsoft.MixedReality.OpenXR;
using Microsoft.MixedReality.SampleQRCodes;

public class StimServer : MonoBehaviour
{
    /* Control the position and frequency of flashing SSVEP stimuli based on
       data received from Python client via TCP/IP. The stimuli are designed to appear
       above a robotic arm gripper that is controlled by the client. Gripper
       coordinates are mapped to HoloLens coordinates using a QR code placed in
       the environment and aligned to the robot. */

    // Comms with Python client
    private int _connectionPort = 25001;
    TcpListener _server;
    TcpClient _client;
    Thread _thread;
    bool _running;
    private bool _quitApp = false; // app quit flag

    // Stimulus
    public float StartTime; // start flashing
    public float[] Freqs = { 0, 0, 0, 0, 0 }; // frequency (Hz) by direction [t, b, l, r, m]
    private float _maxOffset = 0.15f; // maximum stimulus offset from gripper
    public List<Vector3> StimOffsets; // offset of each stimulus from gripper

    // Robot movement
    private Vector3 _gripperStart = new(0.199f, -0.311f, -0.283f); // gripper start position
    private Vector3 _robotToQR = new(-0.182f, 0.378f, 0.414f); // QR position relative to robot
    private Vector3 _gripperPos; // dynamic gripper position relative to robot
    public Pose GripperPose; // gripper pose in HoloLens coordinates

    // QR code tracking for coordinate transformation
    public QRCodesManager QrCodesManager;
    private Microsoft.MixedReality.QR.QRCode _qrCode;
    private Guid _qrCoords;
    public Pose QRCodePose;
    public GameObject QRCodeFrame; // QR frame axes
    public GameObject DummyQR; // visualise dummy QR code

    void SetOffsets(float offset)
        // Set the position of each stimulus relative to the gripper
    {
        StimOffsets[0] = new Vector3(0, 0, offset);  // t
        StimOffsets[1] = new Vector3(0, 0, -offset); // b
        StimOffsets[2] = new Vector3(-offset, 0, 0); // l
        StimOffsets[3] = new Vector3(offset, 0, 0);  // r
    }

    void Start()
        // Setup initial stimulus and comms
    {
        StartTime = Time.time;
        _gripperPos = _gripperStart;
        GripperPose = new Pose(_gripperPos, Quaternion.identity);
        SetOffsets(_maxOffset);

        // visualise dummy QR and start detection
        QRCodePose = new Pose(DummyQR.transform.position, DummyQR.transform.rotation);
        UpdateGripperPose(QRCodePose, _robotToQR, _gripperPos);
        QRCodeFrame.transform.SetPositionAndRotation(GripperPose.position, GripperPose.rotation);
        QrCodesManager.StartQRTracking();

        // receive on a separate thread so Unity doesn't freeze waiting for data
        ThreadStart ts = new(GetData);
        _thread = new Thread(ts);
        _thread.Start();
    }

    void GetData()
        // Get data from Python client via TCP/IP
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
        // Read data from Python and update stimulus
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
            case "setup":
                // set stim offsets
                if (float.TryParse(msgArray[1], out _maxOffset)) SetOffsets(_maxOffset);
                break;

            case "move":
                // move based on new coordinates
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

                break;

            case "quit":
                _quitApp = true;
                break;

        }

        // echo msg as response
        nwStream.Write(buffer, 0, bytesRead);

    }

    void UpdatePos(string data)
        // Update simulated gripper coordinates
    {
        var dataElements = data.Split(',');
        _gripperPos = new Vector3(float.Parse(dataElements[0]), float.Parse(dataElements[1]),
            float.Parse(dataElements[2]));
    }

    void UpdateGripperPose(Pose QRPose, Vector3 RobotOrigin, Vector3 GripperPos)
        // Update gripper rotation and position in HoloLens frame
    {
        var gripperPos = QRPose.position + QRPose.rotation * (RobotOrigin + GripperPos);
        GripperPose = new Pose(gripperPos, QRPose.rotation);
    }

    void Update()
        // look for QR code and setup global coords
    {
        if (_quitApp) Application.Quit();
        if (!QrCodesManager.IsTrackerRunning || (QrCodesManager.GetList().Count == 0)) return;
        // actual QR code found
        _qrCode = QrCodesManager.GetList()[0];
        _qrCoords = _qrCode.SpatialGraphNodeId;
        SpatialGraphNode coordinateSystem = SpatialGraphNode.FromStaticNodeId(_qrCoords);

        if (!coordinateSystem.TryLocate(FrameTime.OnUpdate, out QRCodePose)) return;
        Debug.Log("Found QR coordinates");
        QRCodePose = new Pose(QRCodePose.position, QRCodePose.rotation);
        UpdateGripperPose(QRCodePose, _robotToQR, _gripperPos);

        // stop looking for QR codes and delete dummy shapes
        QrCodesManager.StopQRTracking();
        Destroy(DummyQR);
        Destroy(QRCodeFrame);
    }
}