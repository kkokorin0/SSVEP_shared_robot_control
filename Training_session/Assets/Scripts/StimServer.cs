using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using System.Threading;
using Microsoft.MixedReality.OpenXR;
using Microsoft.MixedReality.SampleQRCodes;
using Microsoft.MixedReality.QR;
using Microsoft.MixedReality.Toolkit.Utilities;

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

    // QR code tracking
    public QRCodesManager QrCodesManager;
    private Microsoft.MixedReality.QR.QRCode _qrCode;
    private Guid _qrCoords;
    public Pose QRCodePose;
    public GameObject QRCodeFrame;

    void Start()
    {
        StartTime = Time.time;
        GripperPos = _gripperStart;

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

    //Get pose in Unity coordinates based on spatial graph node id
    /*private Pose GetPose(Guid spatialGraphNodeId)
    {
        System.Numerics.Matrix4x4? relativePose = System.Numerics.Matrix4x4.Identity;

        SpatialGraphNode coordinateSystem = SpatialGraphNode.FromStaticNodeId(spatialGraphNodeId);

        /*SpatialCoordinateSystem coordinateSystem =
          Windows.Perception.Spatial.Preview.SpatialGraphInteropPreview.
            CreateCoordinateSystemForNode(spatialGraphNodeId);#1#

        SpatialCoordinateSystem rootSpatialCoordinateSystem =
          (SpatialCoordinateSystem)System.Runtime.InteropServices.Marshal.
              GetObjectForIUnknown(WorldManager.GetNativeISpatialCoordinateSystemPtr());

        // Get the relative transform from the unity origin
        relativePose = coordinateSystem.TryGetTransformTo(rootSpatialCoordinateSystem);

        System.Numerics.Matrix4x4 newMatrix = relativePose.Value;

        // Platform coordinates are all right handed and unity uses left handed matrices. 
        // so we convert the matrix from rhs-rhs to lhs-lhs 
        newMatrix.M13 = -newMatrix.M13;
        newMatrix.M23 = -newMatrix.M23;
        newMatrix.M43 = -newMatrix.M43;

        newMatrix.M31 = -newMatrix.M31;
        newMatrix.M32 = -newMatrix.M32;
        newMatrix.M34 = -newMatrix.M34;

        System.Numerics.Vector3 scale;
        System.Numerics.Quaternion rotation1;
        System.Numerics.Vector3 translation1;

        System.Numerics.Matrix4x4.Decompose(newMatrix, out scale, out rotation1,
                                            out translation1);
        var translation = new Vector3(translation1.X, translation1.Y, translation1.Z);
        var rotation = new Quaternion(rotation1.X, rotation1.Y, rotation1.Z, rotation1.W);
        var pose = new Pose(translation, rotation);

        // If there is a parent to the camera that means we are using teleport and we
        // should not apply the teleport to these objects so apply the inverse
        if (CameraCache.Main.transform.parent != null)
        {
            pose = pose.GetTransformedBy(CameraCache.Main.transform.parent);
        }

        return pose;
    }*/

    void Update()
    {
        // look for QR code and setup global coords
        if (QrCodesManager.GetList().Count == 0) return;
        _qrCode = QrCodesManager.GetList()[0];
        _qrCoords = _qrCode.SpatialGraphNodeId;
        SpatialGraphNode coordinateSystem = SpatialGraphNode.FromStaticNodeId(_qrCoords);
        coordinateSystem.TryLocate(FrameTime.OnUpdate, out Pose QRCodePose);
        QRCodeFrame.transform.SetPositionAndRotation(QRCodePose.position, QRCodePose.rotation);

        /*Debug.Log(QrCodesManager.GetList()[0].ToString());*/
        /*QrCodesManager.StopQRTracking();*/

    }
}