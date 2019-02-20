using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

namespace PA_DronePack_Free
{
	public class DroneAgent : Agent
	{
		public PA_DroneController dcoScript;

		public GameObject goal;
		Vector3 dronInitPos;

		float preDist;
		float curDist;

		float act0;
		float act1;
		float act2;

		public override void InitializeAgent()
		{
			dcoScript = GetComponent<PA_DroneController>();
			dronInitPos = gameObject.transform.position;
		}

		public override void CollectObservations()
		{
			AddVectorObs(gameObject.transform.position - goal.transform.position);
			AddVectorObs(gameObject.GetComponent<Rigidbody>().velocity);
		}

		public override void AgentAction(float[] vectorAction, string textAction)
		{
			act0 = Mathf.Clamp(vectorAction[0], -1f, 1f);
			act1 = Mathf.Clamp(vectorAction[1], -1f, 1f);
			act2 = Mathf.Clamp(vectorAction[2], -1f, 1f);

			dcoScript.DriveInput(act0);
			dcoScript.StrafeInput(act1);
			dcoScript.LiftInput(act2);

			if ((goal.transform.position - gameObject.transform.position).magnitude < 0.5f)
			{
				SetReward(5);
				Done();
				//Debug.Log("Success.");
			}
			else if ((goal.transform.position - gameObject.transform.position).magnitude > 6f)
			{
				SetReward(-5);
				Done();
				//Debug.Log("Failed.");

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
			gameObject.transform.position = dronInitPos;
			gameObject.GetComponent<Rigidbody>().velocity = Vector3.zero;
			goal.transform.position = gameObject.transform.position + new Vector3(Random.Range(-5f, 5f), Random.Range(-5f, 5f), Random.Range(-5f, 5f));
			preDist = (goal.transform.position - gameObject.transform.position).magnitude;
		}

	}
}