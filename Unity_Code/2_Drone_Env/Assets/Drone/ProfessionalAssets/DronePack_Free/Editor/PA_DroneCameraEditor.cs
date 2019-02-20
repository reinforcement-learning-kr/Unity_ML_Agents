using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

namespace PA_DronePack_Free
{
    [CustomEditor(typeof(PA_DroneCamera))]
    [CanEditMultipleObjects]
    public class PA_DroneCameraEditor : Editor
    {
        #region Varibles
        PA_DroneCamera dcScript;
        #endregion

        public void OnEnable()
        {
            #region Script Targets
            dcScript = (PA_DroneCamera)target;
            #endregion
        }

        public override void OnInspectorGUI()
        {
            #region Edit Script
            GUI.enabled = false;
            EditorGUILayout.ObjectField("Script", MonoScript.FromMonoBehaviour(dcScript), typeof(PA_DroneCamera), false);
            GUI.enabled = true;
            #endregion

            #region Main Settings
            EditorGUILayout.LabelField("Main Settings", EditorStyles.boldLabel);
            SerializedProperty cameraMode = serializedObject.FindProperty("cameraMode");
            EditorGUILayout.PropertyField(cameraMode);
            SerializedProperty followMode = serializedObject.FindProperty("followMode");
            EditorGUILayout.PropertyField(followMode);
            if (dcScript.followMode == PA_DroneCamera.FollowMode.smooth)
            {
                SerializedProperty followSmoothing = serializedObject.FindProperty("followSmoothing");
                EditorGUILayout.PropertyField(followSmoothing);
            }
            GUILayout.Space(10f);
            #endregion

            #region TPS Settings
            EditorGUILayout.LabelField("TPS Settings", EditorStyles.boldLabel);
            dcScript.findTarget = EditorGUILayout.Toggle("Auto Target?", dcScript.findTarget);
            if (!dcScript.findTarget)
            {
                SerializedProperty target = serializedObject.FindProperty("target");
                EditorGUILayout.PropertyField(target);
                GUILayout.Space(10f);
            }
            dcScript.autoPosition = EditorGUILayout.Toggle("Auto Position?", dcScript.autoPosition);
            if (!dcScript.autoPosition)
            {
                SerializedProperty height = serializedObject.FindProperty("height");
                EditorGUILayout.PropertyField(height);
                SerializedProperty distance = serializedObject.FindProperty("distance");
                EditorGUILayout.PropertyField(distance);
                SerializedProperty angle = serializedObject.FindProperty("angle");
                EditorGUILayout.PropertyField(angle);
                GUILayout.Space(10f);
            }
            dcScript.freeLook = EditorGUILayout.Toggle("Free Look?", dcScript.freeLook);
            SerializedProperty xSensivity = serializedObject.FindProperty("xSensivity");
            EditorGUILayout.PropertyField(xSensivity);
            SerializedProperty ySensivity = serializedObject.FindProperty("ySensivity");
            EditorGUILayout.PropertyField(ySensivity);
            SerializedProperty invertYAxis = serializedObject.FindProperty("invertYAxis");
            EditorGUILayout.PropertyField(invertYAxis);
            GUILayout.Space(10f);
            #endregion

            #region FPS Settings
            EditorGUILayout.LabelField("FPS Settings", EditorStyles.boldLabel);
            dcScript.findFPS = EditorGUILayout.Toggle("Auto Target?", dcScript.findFPS);
            if (!dcScript.findFPS)
            {
                SerializedProperty fpsPosition = serializedObject.FindProperty("fpsPosition");
                EditorGUILayout.PropertyField(fpsPosition);
                GUILayout.Space(10f);
            }
            dcScript.gyroscopeEnabled = EditorGUILayout.Toggle("Use Gyroscope?", dcScript.gyroscopeEnabled);
            GUILayout.Space(10f);

            EditorGUILayout.LabelField("Other Settings", EditorStyles.boldLabel);
            SerializedProperty jitterRigidBodies = serializedObject.FindProperty("jitterRigidBodies");
            EditorGUILayout.PropertyField(jitterRigidBodies, true);
            #endregion

            #region Finalize editor changes
            if (GUI.changed) { serializedObject.ApplyModifiedProperties(); } // any changes we made to serialized objects will be finalized here
            EditorUtility.SetDirty(dcScript);
            #endregion
        }
    }
}
