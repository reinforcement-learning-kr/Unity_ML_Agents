using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

namespace PA_DronePack_Free
{
    [CustomEditor(typeof(PA_DroneAxisInput))]
    [CanEditMultipleObjects]
    public class PA_DroneAxisInputEditor : Editor
    {
        #region varibles
        PA_DroneAxisInput daiScript;
        #endregion

        public void OnEnable()
        {
            #region targets
            daiScript = (PA_DroneAxisInput)target;
            #endregion
        }

        public override void OnInspectorGUI()
        {
            #region Edit Script
            GUI.enabled = false;
            EditorGUILayout.ObjectField("Script", MonoScript.FromMonoBehaviour(daiScript), typeof(PA_DroneAxisInput), false);
            GUI.enabled = true;
            #endregion

            #region Input Type
            SerializedProperty inputType = serializedObject.FindProperty("inputType");
            EditorGUILayout.PropertyField(inputType);
            GUILayout.Space(10f);
            #endregion

            #region Desktop Input
            if (daiScript.inputType == PA_DroneAxisInput.InputType.Desktop)
            {
                EditorGUILayout.LabelField("Input Axis", EditorStyles.boldLabel);
                EditorGUI.BeginDisabledGroup(true);
                daiScript.forwardBackward = EditorGUILayout.TextField("Forward & Backward", "Vertical");
                daiScript.strafeLeftRight = EditorGUILayout.TextField("Strafe Left & Right", "Horizontal");
                daiScript.riseLower = EditorGUILayout.TextField("Rise & Lower", "Lift");
                daiScript.turn = EditorGUILayout.TextField("Turn", "Mouse X");
                GUILayout.Space(10f);
                daiScript.cameraRiseLower = EditorGUILayout.TextField("Camera Rise & Lower", "Mouse Y");
                daiScript.cameraTurn = EditorGUILayout.TextField("Camera Turn", "Mouse X");
                EditorGUI.EndDisabledGroup();
                GUILayout.Space(10f);

                EditorGUILayout.LabelField("Input Axis / Button / Keycode", EditorStyles.boldLabel);
                EditorGUI.BeginDisabledGroup(true);
                daiScript.toggleMotor = EditorGUILayout.TextField("Toggle Motor", "Z");
                daiScript.toggleCameraMode = EditorGUILayout.TextField("Change Camera Mode", "C");
                daiScript.toggleCameraGyro = EditorGUILayout.TextField("Toggle Camera Gyro", "G");
                daiScript.toggleFollowMode = EditorGUILayout.TextField("Change Follow Mode", "F");
                daiScript.cameraFreeLook = EditorGUILayout.TextField("Hold FreeLook", "LeftAlt");
                EditorGUI.EndDisabledGroup();
            }
            #endregion

            #region Gamepad Input
            if (daiScript.inputType == PA_DroneAxisInput.InputType.Gamepad)
            {
                EditorGUILayout.LabelField("Input Axis", EditorStyles.boldLabel);
                EditorGUI.BeginDisabledGroup(true);
                daiScript.forwardBackward = EditorGUILayout.TextField("Forward & Backward", "GP SecondaryJoystick Y");
                daiScript.strafeLeftRight = EditorGUILayout.TextField("Strafe Left & Right", "GP SecondaryJoystick X");
                daiScript.riseLower = EditorGUILayout.TextField("Rise & Lower", "GP PrimaryJoystick Y");
                daiScript.turn = EditorGUILayout.TextField("Turn", "GP PrimaryJoystick X");
                GUILayout.Space(10f);
                daiScript.cameraRiseLower = EditorGUILayout.TextField("Camera Rise & Lower", "GP DPad Y");
                daiScript.cameraTurn = EditorGUILayout.TextField("Camera Turn", "GP PrimaryJoystick X");
                EditorGUI.EndDisabledGroup();
                GUILayout.Space(10f);

                EditorGUILayout.LabelField("Input Axis / Button", EditorStyles.boldLabel);
                EditorGUI.BeginDisabledGroup(true);
                daiScript.toggleMotor = EditorGUILayout.TextField("Toggle Motor", "GP Button 0");
                daiScript.toggleCameraMode = EditorGUILayout.TextField("Change Camera Mode", "GP Button 1");
                daiScript.toggleCameraGyro = EditorGUILayout.TextField("Toggle Camera Gyro", "GP Button 2");
                daiScript.toggleFollowMode = EditorGUILayout.TextField("Change Follow Mode", "GP Button 3");
                daiScript.cameraFreeLook = EditorGUILayout.TextField("Hold FreeLook", "GP Button 5");
                EditorGUI.EndDisabledGroup();
            }
            #endregion

            #region OpenVR Input
            if (daiScript.inputType == PA_DroneAxisInput.InputType.OpenVR)
            {
                EditorGUILayout.LabelField("Input Axis", EditorStyles.boldLabel);
                EditorGUI.BeginDisabledGroup(true);
                daiScript.forwardBackward = EditorGUILayout.TextField("Forward & Backward", "OVR RightJoystick Y");
                daiScript.strafeLeftRight = EditorGUILayout.TextField("Strafe Left & Right", "OVR RightJoystick X");
                daiScript.riseLower = EditorGUILayout.TextField("Rise & Lower", "OVR LeftJoystick Y");
                daiScript.turn = EditorGUILayout.TextField("Turn", "OVR LeftJoystick X");
                GUILayout.Space(10f);
                daiScript.cameraRiseLower = EditorGUILayout.TextField("Camera Rise & Lower", "");
                daiScript.cameraTurn = EditorGUILayout.TextField("Camera Turn", "OVR LeftJoystick X");
                EditorGUI.EndDisabledGroup();
                GUILayout.Space(10f);

                EditorGUILayout.LabelField("Input Axis / Button", EditorStyles.boldLabel);
                EditorGUI.BeginDisabledGroup(true);
                daiScript.toggleMotor = EditorGUILayout.TextField("Toggle Motor", "OVR RightButton 0");
                daiScript.toggleCameraMode = EditorGUILayout.TextField("Change Camera Mode", "");
                daiScript.toggleCameraGyro = EditorGUILayout.TextField("Toggle Camera Gyro", "");
                daiScript.toggleFollowMode = EditorGUILayout.TextField("Change Follow Mode", "");
                daiScript.cameraFreeLook = EditorGUILayout.TextField("Hold FreeLook", "OVR RightTrigger");
                EditorGUI.EndDisabledGroup();
            }
            #endregion

            #region Custom Input
            if (daiScript.inputType == PA_DroneAxisInput.InputType.Custom)
            {
                EditorGUILayout.LabelField("Input Axis", EditorStyles.boldLabel);
                daiScript.cForwardBackward = EditorGUILayout.TextField("Forward & Backward", daiScript.cForwardBackward);
                daiScript.cStrafeLeftRight = EditorGUILayout.TextField("Strafe Left & Right", daiScript.cStrafeLeftRight);
                daiScript.cRiseLower = EditorGUILayout.TextField("Rise & Lower", daiScript.cRiseLower);
                daiScript.cTurn = EditorGUILayout.TextField("Turn", daiScript.cTurn);
                GUILayout.Space(10f);
                daiScript.cCameraRiseLower = EditorGUILayout.TextField("Camera Rise & Lower", daiScript.cCameraRiseLower);
                daiScript.cCameraTurn = EditorGUILayout.TextField("Camera Turn", daiScript.cCameraTurn);
                GUILayout.Space(10f);

                EditorGUILayout.LabelField("Input Axis / Button / Keycode", EditorStyles.boldLabel);
                daiScript.cToggleMotor = EditorGUILayout.TextField("Toggle Motor", daiScript.cToggleMotor);
                daiScript.cToggleCameraMode = EditorGUILayout.TextField("Change Camera Mode", daiScript.cToggleCameraMode);
                daiScript.cToggleCameraGyro = EditorGUILayout.TextField("Toggle Camera Gyro", daiScript.cToggleCameraGyro);
                daiScript.cToggleFollowMode = EditorGUILayout.TextField("Change Follow Mode", daiScript.cToggleFollowMode);
                daiScript.cCameraFreeLook = EditorGUILayout.TextField("Hold FreeLook", daiScript.cCameraFreeLook);
            }
            #endregion

            #region finalize editor changes
            if (GUI.changed) { serializedObject.ApplyModifiedProperties(); } // any changes we made to serialized objects will be finalized here
            EditorUtility.SetDirty(daiScript);
            #endregion
        }
    }
}