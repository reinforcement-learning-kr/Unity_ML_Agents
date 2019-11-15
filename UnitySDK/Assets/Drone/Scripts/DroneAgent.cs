using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

namespace PA_DronePack
{ 
	public class DroneAgent : Agent
	{
		private PA_DroneController dcoScript;
		public GameObject goal;
		Vector3 droneInitPos;
		Quaternion droneInitRot;
		float preDist, curDist;

		public override void InitializeAgent()
		{
			dcoScript = GetComponent<PA_DroneController>();
			droneInitPos = gameObject.transform.position;
			droneInitRot = gameObject.transform.rotation;
		}

		public override void CollectObservations()
		{
			AddVectorObs(gameObject.transform.position - goal.transform.position);
			AddVectorObs(gameObject.GetComponent<Rigidbody>().velocity);
			AddVectorObs(gameObject.GetComponent<Rigidbody>().angularVelocity);
		}
		public override void AgentAction(float[] vectorAction, string textAction)
		{
			var act0 = Mathf.Clamp(vectorAction[0], -1f, 1f);
			var act1 = Mathf.Clamp(vectorAction[1], -1f, 1f);
			var act2 = Mathf.Clamp(vectorAction[2], -1f, 1f);

			dcoScript.DriveInput(act0);
			dcoScript.StrafeInput(act1);
			dcoScript.LiftInput(act2);

			if ((goal.transform.position - gameObject.transform.position).magnitude < 0.5f)
			{
				SetReward(1);
				Done();
			}
			else if ((goal.transform.position - gameObject.transform.position).magnitude > 10f)
			{
				SetReward(-1);
				Done();
			}
			else
			{
				curDist = (goal.transform.position - gameObject.transform.position).magnitude;
				var reward = (preDist - curDist);
				SetReward(reward);
				preDist = curDist;
			}
		}
		public override void AgentReset()
		{
			gameObject.GetComponent<Rigidbody>().velocity = Vector3.zero;
			gameObject.GetComponent<Rigidbody>().angularVelocity = Vector3.zero;
			gameObject.transform.position = droneInitPos;
			gameObject.transform.rotation = droneInitRot;
			goal.transform.position = gameObject.transform.position + new Vector3(Random.Range(-5f, 5f),
																					Random.Range(-5f, 5f),
																					Random.Range(-5f, 5f));
			preDist = (goal.transform.position - gameObject.transform.position).magnitude;
		}
	}
}
