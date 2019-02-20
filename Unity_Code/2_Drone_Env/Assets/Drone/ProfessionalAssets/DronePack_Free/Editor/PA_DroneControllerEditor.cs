using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

namespace PA_DronePack_Free
{
    
    [CustomEditor(typeof(PA_DroneController))]
    [CanEditMultipleObjects]
    public class PA_DroneControllerEditor : Editor
    {
        #region varibles
        PA_DroneController dcoScript;
        #endregion

        public void OnEnable()
        {
            #region targets
            dcoScript = (PA_DroneController)target;
            #endregion
        }

        public override void OnInspectorGUI()
        {
            #region Edit Script
            GUI.enabled = false;
            EditorGUILayout.ObjectField("Script", MonoScript.FromMonoBehaviour(dcoScript), typeof(PA_DroneController), false);
            GUI.enabled = true;
            #endregion

            #region Movement
            EditorGUILayout.LabelField("Movement Values", EditorStyles.boldLabel);

            dcoScript.forwardSpeed = EditorGUILayout.FloatField("Forward Speed", dcoScript.forwardSpeed);
            dcoScript.backwardSpeed = EditorGUILayout.FloatField("Backward Speed", dcoScript.backwardSpeed);
            GUILayout.Space(10f);

            dcoScript.rightSpeed = EditorGUILayout.FloatField("Strafe Right Speed", dcoScript.rightSpeed);
            dcoScript.leftSpeed = EditorGUILayout.FloatField("Strafe Left Speed", dcoScript.leftSpeed);
            GUILayout.Space(10f);

            dcoScript.riseSpeed = EditorGUILayout.FloatField("Height Rise Speed", dcoScript.riseSpeed);
            dcoScript.lowerSpeed = EditorGUILayout.FloatField("Height Lower Speed", dcoScript.lowerSpeed);
            GUILayout.Space(10f);

            dcoScript.acceleration = EditorGUILayout.Slider("Acceleration", dcoScript.acceleration, 0.01f, 1f);
            dcoScript.deceleration = EditorGUILayout.Slider("Deceleration", dcoScript.deceleration, 0.01f, 1f);
            dcoScript.stability = EditorGUILayout.Slider("Stability", dcoScript.stability, 0.01f, 0.3f);
            dcoScript.turnSensitivty = EditorGUILayout.Slider("Turn Sensitivity", dcoScript.turnSensitivty, 0.1f, 5f);
            GUILayout.Space(10f);

            dcoScript.motorOn = EditorGUILayout.Toggle("Is Motor On?", dcoScript.motorOn);
            GUILayout.Space(10f);
            #endregion

            #region Apperance
            EditorGUILayout.LabelField("Appearance", EditorStyles.boldLabel);
            EditorGUI.BeginDisabledGroup(true);
            EditorGUILayout.HelpBox("Customizable in Full-Version", MessageType.Info);
            SerializedProperty propellers = serializedObject.FindProperty("propellers");
            EditorGUILayout.PropertyField(propellers, true);
            SerializedProperty propSpinSpeed = serializedObject.FindProperty("propSpinSpeed");
            EditorGUILayout.PropertyField(propSpinSpeed);
            SerializedProperty propStopSpeed = serializedObject.FindProperty("propStopSpeed");
            EditorGUILayout.PropertyField(propStopSpeed);
            SerializedProperty frontTilt = serializedObject.FindProperty("frontTilt");
            EditorGUILayout.PropertyField(frontTilt);
            SerializedProperty backTilt = serializedObject.FindProperty("backTilt");
            EditorGUILayout.PropertyField(backTilt);
            SerializedProperty rightTilt = serializedObject.FindProperty("rightTilt");
            EditorGUILayout.PropertyField(rightTilt);
            SerializedProperty leftTilt = serializedObject.FindProperty("leftTilt");
            EditorGUILayout.PropertyField(leftTilt);
            EditorGUI.EndDisabledGroup();
            GUILayout.Space(10f);
            #endregion

            #region Collision Settings
            EditorGUILayout.LabelField("Collision Settings", EditorStyles.boldLabel);
            dcoScript.fallAfterCollision = EditorGUILayout.Toggle("Fall After Collision?", dcoScript.fallAfterCollision);
            EditorGUI.BeginDisabledGroup(true);
            EditorGUILayout.HelpBox("Customizable in Full-Version", MessageType.Info);
            SerializedProperty fallMinimumForce = serializedObject.FindProperty("fallMinimumForce");
            EditorGUILayout.PropertyField(fallMinimumForce);
            SerializedProperty sparkMinimumForce = serializedObject.FindProperty("sparkMinimumForce");
            EditorGUILayout.PropertyField(sparkMinimumForce);
            SerializedProperty sparkPrefab = serializedObject.FindProperty("sparkPrefab");
            EditorGUILayout.PropertyField(sparkPrefab);
            EditorGUI.EndDisabledGroup();
            GUILayout.Space(10f);
            #endregion

            #region Sound effects
            EditorGUILayout.LabelField("Sound Effects", EditorStyles.boldLabel);
            EditorGUI.BeginDisabledGroup(true);
            EditorGUILayout.HelpBox("Customizable in Full-Version", MessageType.Info);
            SerializedProperty flyingSound = serializedObject.FindProperty("flyingSound");
            EditorGUILayout.PropertyField(flyingSound);
            SerializedProperty sparkSound = serializedObject.FindProperty("sparkSound");
            EditorGUILayout.PropertyField(sparkSound);
            EditorGUI.EndDisabledGroup();
            GUILayout.Space(10f);
            #endregion

            #region Read Only variables
            EditorGUILayout.LabelField("Read Only Variables", EditorStyles.boldLabel);
            EditorGUI.BeginDisabledGroup(true);
            EditorGUILayout.HelpBox("Available in Full-Version", MessageType.Info);
            EditorGUILayout.TextField("Collision Magnitude", "NaN");
            EditorGUILayout.TextField("Lift Force", "NaN");
            EditorGUILayout.TextField("Drive Force", "NaN");
            EditorGUILayout.TextField("Strafe Force", "NaN");
            EditorGUILayout.TextField("Turn Force", "NaN");
            EditorGUILayout.TextField("Ground Distance", "NaN");
            EditorGUILayout.TextField("Upright Angle Distance", "NaN");
            EditorGUILayout.TextField("Current Prop Speed", "NaN");
            EditorGUILayout.Vector3Field("Start Position", Vector3.zero);
            EditorGUILayout.Vector3Field("Start Rotation", Vector3.zero);
            EditorGUI.EndDisabledGroup();
            #endregion

            #region finalize editor changes
            if (GUI.changed) { serializedObject.ApplyModifiedProperties(); }
            EditorUtility.SetDirty(dcoScript);
            #endregion
        }
    }
}
