using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class DroneAgent : Agent
{
	public GameObject goal;
	Vector3 droneInitPos;
	float preDist, curDist;

	public override void InitializeAgent()
	{
		droneInitPos = gameObject.transform.position;
	}

	public override void CollectObservations()
	{
		AddVectorObs((goal.transform.position - gameObject.transform.position).normalized);
	}

	public override void AgentAction(float[] vectorAction, string textAction)
	{
		var act0 = vectorAction[0];
		var act1 = vectorAction[1];
		var act2 = vectorAction[2];

		Vector3 newPos = new Vector3(gameObject.transform.position[0] + act0,
										gameObject.transform.position[1] + act1,
										gameObject.transform.position[2] + act2);

		gameObject.transform.position = newPos;

		if ((goal.transform.position - gameObject.transform.position).magnitude < 0.5f)
		{
			SetReward(1);
			Done();
		}
		else if ((goal.transform.position - gameObject.transform.position).magnitude > 6f)
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
		gameObject.transform.position = droneInitPos;
		goal.transform.position = gameObject.transform.position + new Vector3(Random.Range(-5f, 5f),
																				Random.Range(-5f, 5f),
																				Random.Range(-5f, 5f));
		preDist = (goal.transform.position - gameObject.transform.position).magnitude;
	}

}