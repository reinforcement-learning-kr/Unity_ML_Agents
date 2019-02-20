using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace PA_DronePack_Free
{
    public class PA_DroneAxisInput : MonoBehaviour
    {
        public enum InputType { Desktop, Gamepad, OpenVR, Custom }
        [HideInInspector]
        public InputType inputType = InputType.Desktop;

        [HideInInspector]
        public string forwardBackward;
        [HideInInspector]
        public string strafeLeftRight;
        [HideInInspector]
        public string riseLower;
        [HideInInspector]
        public string turn;

        [HideInInspector]
        public string toggleMotor;
        [HideInInspector]
        public string toggleCameraMode;
        [HideInInspector]
        public string toggleCameraGyro;
        [HideInInspector]
        public string toggleFollowMode;

        [HideInInspector]
        public string cameraRiseLower;
        [HideInInspector]
        public string cameraTurn;
        [HideInInspector]
        public string cameraFreeLook;

        public string cForwardBackward;
        public string cStrafeLeftRight;
        public string cRiseLower;
        public string cTurn;

        public string cCameraRiseLower;
        public string cCameraTurn;
        public string cCameraFreeLook;

        public string cToggleMotor;
        public string cToggleCameraMode;
        public string cToggleCameraGyro;
        public string cToggleFollowMode;

        [HideInInspector]
        public PA_DroneController dcoScript;
        [HideInInspector]
        public PA_DroneCamera dcScript;

        private bool toggleMotorIsKey = false;
        private bool toggleMotorIsAxis = false;

        private bool toggleCameraModeIsKey = false;
        private bool toggleCameraModeIsAxis = false;

        private bool toggleCameraGyroIsKey = false;
        private bool toggleCameraGyroIsAxis = false;

        private bool toggleFollowModeIsKey = false;
        private bool toggleFollowModeIsAxis = false;

        private bool cameraFreeLookIsKey = false;
        private bool cameraFreeLookIsAxis = false;

        void Awake()
        {
            dcoScript = GetComponent<PA_DroneController>();
            dcScript = FindObjectOfType<PA_DroneCamera>();
            if (inputType == InputType.Custom)
            {
                forwardBackward = cForwardBackward;
                strafeLeftRight = cStrafeLeftRight;
                riseLower = cRiseLower;
                turn = cTurn;
                toggleMotor = cToggleMotor;
                toggleCameraMode = cToggleCameraMode;
                toggleCameraGyro = cToggleCameraGyro;
                toggleFollowMode = cToggleFollowMode;
                cameraRiseLower = cCameraRiseLower;
                cameraTurn = cCameraTurn;
                cameraFreeLook = cCameraFreeLook;
            }
        }

        void Start()
        {
            ValidateInputs();
        }

        void Update()
        {
            if (forwardBackward != "")
            {
                dcoScript.DriveInput(Input.GetAxisRaw(forwardBackward));
            }

            if (strafeLeftRight != "")
            {
                dcoScript.StrafeInput(Input.GetAxisRaw(strafeLeftRight));
            }

            if (riseLower != "")
            {
                dcoScript.LiftInput(Input.GetAxisRaw(riseLower));
            }

            if (turn != "")
            {
                dcoScript.TurnInput(Input.GetAxis(turn));
            }

            if(cameraRiseLower != "")
            {
                dcScript.LiftInput(Input.GetAxis(cameraRiseLower));
            }

            if (cameraTurn != "")
            {
                dcScript.TurnInput(Input.GetAxis(cameraTurn));
            }

            if (toggleMotor != "")
            {
                if (toggleMotorIsKey)
                {
                    if (Input.GetKeyDown((KeyCode)System.Enum.Parse(typeof(KeyCode), toggleMotor))) { dcoScript.ToggleMotor(); }
                }
                if (toggleMotorIsAxis)
                {
                    if (Input.GetButtonDown(toggleMotor)) { dcoScript.ToggleMotor(); }
                }
            }

            if (toggleCameraMode != "" && dcScript)
            {
                if (toggleCameraModeIsKey)
                {
                    if (Input.GetKeyDown((KeyCode)System.Enum.Parse(typeof(KeyCode), toggleCameraMode))) { dcScript.ChangeCameraMode(); }
                }
                if (toggleCameraModeIsAxis)
                {
                    if (Input.GetButtonDown(toggleCameraMode)) { dcScript.ChangeCameraMode(); }
                }
            }

            if (toggleCameraGyro != "" && dcScript)
            {
                if (toggleCameraGyroIsKey)
                {
                    if (Input.GetKeyDown((KeyCode)System.Enum.Parse(typeof(KeyCode), toggleCameraGyro))) { dcScript.ChangeGyroscope(); }
                }
                if (toggleCameraGyroIsAxis)
                {
                    if (Input.GetButtonDown(toggleCameraGyro)) { dcScript.ChangeGyroscope(); }
                }
            }

            if (toggleFollowMode != "" && dcScript)
            {
                if (toggleFollowModeIsKey)
                {
                    if (Input.GetKeyDown((KeyCode)System.Enum.Parse(typeof(KeyCode), toggleFollowMode))) { dcScript.ChangeFollowMode(); }
                }
                if (toggleFollowModeIsAxis)
                {
                    if (Input.GetButtonDown(toggleFollowMode)) { dcScript.ChangeFollowMode(); }
                }
            }

            if (cameraFreeLook != "" && dcScript)
            {
                if (cameraFreeLookIsKey)
                {
                    if (Input.GetKeyDown((KeyCode)System.Enum.Parse(typeof(KeyCode), cameraFreeLook))) { dcScript.CanFreeLook(true); }
                    if (Input.GetKeyUp((KeyCode)System.Enum.Parse(typeof(KeyCode), cameraFreeLook))) { dcScript.CanFreeLook(false); }
                }
                if (cameraFreeLookIsAxis)
                {
                    if (Input.GetButtonDown(cameraFreeLook)) { dcScript.CanFreeLook(true); }
                    if (Input.GetButtonUp(cameraFreeLook)) { dcScript.CanFreeLook(false); }
                }
            }
        }

        void ValidateInputs()
        {
            if (toggleMotor != "")
            {
                if (ValidKey(toggleMotor)) { toggleMotorIsKey = true; }
                if (ValidAxis(toggleMotor)) { toggleMotorIsAxis = true; }
                if (!toggleMotorIsKey && !toggleMotorIsAxis) { Debug.LogWarning("PA_DroneAxisInput : '" + toggleMotor + "' is not a valid Keycode or Input Axis, it will not be used"); }
            }

            if (toggleCameraMode != "")
            {
                if (ValidKey(toggleCameraMode)) { toggleCameraModeIsKey = true; }
                if (ValidAxis(toggleCameraMode)) { toggleCameraModeIsAxis = true; }
                if (!toggleCameraModeIsKey && !toggleCameraModeIsAxis) { Debug.LogWarning("PA_DroneAxisInput : '" + toggleCameraMode + "' is not a valid Keycode or Input Axis, it will not be used"); }
            }

            if (toggleCameraGyro != "")
            {
                if (ValidKey(toggleCameraGyro)) { toggleCameraGyroIsKey = true; }
                if (ValidAxis(toggleCameraGyro)) { toggleCameraGyroIsAxis = true; }
                if (!toggleCameraGyroIsKey && !toggleCameraGyroIsAxis) { Debug.LogWarning("PA_DroneAxisInput : '" + toggleCameraGyro + "' is not a valid Keycode or Input Axis, it will not be used"); }
            }

            if (toggleFollowMode != "")
            {
                if (ValidKey(toggleFollowMode)) { toggleFollowModeIsKey = true; }
                if (ValidAxis(toggleFollowMode)) { toggleFollowModeIsAxis = true; }
                if (!toggleFollowModeIsKey && !toggleFollowModeIsAxis) { Debug.LogWarning("PA_DroneAxisInput : '" + toggleFollowMode + "' is not a valid Keycode or Input Axis, it will not be used"); }
            }

            if (cameraFreeLook != "")
            {
                if (ValidKey(cameraFreeLook)) { cameraFreeLookIsKey = true; }
                if (ValidAxis(cameraFreeLook)) { cameraFreeLookIsAxis = true; }
                if (!cameraFreeLookIsKey && !cameraFreeLookIsAxis) { Debug.LogWarning("PA_DroneAxisInput : '" + cameraFreeLook + "' is not a valid Keycode or Input Axis, it will not be used"); }
            }
        }

        bool ValidKey(string btnName)
        {
            try { Input.GetKey((KeyCode)System.Enum.Parse(typeof(KeyCode), btnName)); return true; } catch { return false; }
        }

        bool ValidAxis(string axsName)
        {
            try { Input.GetAxis(axsName); return true; } catch { return false; }
        }

    }
}
