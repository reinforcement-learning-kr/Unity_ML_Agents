using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class TwoLegAgent : Agent {
    public GameObject foot;
    public GameObject leg1;
    public GameObject leg2;
    public GameObject body;

    private Rigidbody footRB; 
    private Rigidbody leg1RB;
    private Rigidbody leg2RB;
    private Rigidbody bodyRB;

    private Vector3 body_init_pos;
    private Vector3 foot_init_pos;
    private Vector3 leg1_init_pos;
    private Vector3 leg2_init_pos;

    private Quaternion body_init_rot;
    private Quaternion foot_init_rot;
    private Quaternion leg1_init_rot;
    private Quaternion leg2_init_rot;

    public GameObject ofoot;
    public GameObject oleg1;
    public GameObject oleg2;

    private float pos;

    public override void InitializeAgent()
    {
        footRB = foot.GetComponent<Rigidbody>();
        leg1RB = leg1.GetComponent<Rigidbody>();
        leg2RB = leg2.GetComponent<Rigidbody>();
        bodyRB = body.GetComponent<Rigidbody>();

        body_init_pos = body.transform.position;
        foot_init_pos = foot.transform.position;
        leg1_init_pos = leg1.transform.position;
        leg2_init_pos = leg2.transform.position;

        body_init_rot = body.transform.rotation;
        foot_init_rot = foot.transform.rotation;
        leg1_init_rot = leg1.transform.rotation;
        leg2_init_rot = leg2.transform.rotation;

        bodyRB.velocity = Vector3.zero;
        footRB.velocity = Vector3.zero;
        leg1RB.velocity = Vector3.zero;
        leg2RB.velocity = Vector3.zero;
    }

    public override void CollectObservations()
    {

        AddVectorObs(bodyRB.velocity.x);
        AddVectorObs(bodyRB.velocity.y);
        AddVectorObs(bodyRB.angularVelocity.z);

        AddVectorObs(body.transform.position.x - foot.transform.position.x);
        AddVectorObs(body.transform.position.y - foot.transform.position.y);
        AddVectorObs(body.transform.position.z - foot.transform.position.z);
        AddVectorObs(footRB.velocity.x);
        AddVectorObs(footRB.velocity.y);
        AddVectorObs(footRB.angularVelocity.z);

        AddVectorObs(body.transform.position.x - leg1.transform.position.x);
        AddVectorObs(body.transform.position.y - leg1.transform.position.y);
        AddVectorObs(body.transform.position.z - leg1.transform.position.z);
        AddVectorObs(leg1RB.velocity.x);
        AddVectorObs(leg1RB.velocity.y);
        AddVectorObs(leg1RB.angularVelocity.z);

        AddVectorObs(body.transform.position.x - leg2.transform.position.x);
        AddVectorObs(body.transform.position.y - leg2.transform.position.y);
        AddVectorObs(body.transform.position.z - leg2.transform.position.z);
        AddVectorObs(leg2RB.velocity.x);
        AddVectorObs(leg2RB.velocity.y);
        AddVectorObs(leg2RB.angularVelocity.z);

        AddVectorObs(ofoot.transform.position - foot.transform.position);
        AddVectorObs(oleg1.transform.position - leg1.transform.position);
        AddVectorObs(oleg2.transform.position - leg2.transform.position);
    }

    public override void AgentAction(float[] vectorAction, string textAction)
    {
        for (int k=0; k < vectorAction.Length; k++)
        {
            vectorAction[k] = Mathf.Clamp(vectorAction[k], -1f, 1f);
        }

        float torque = 20f;

        footRB.AddTorque(transform.forward * 2 * torque * vectorAction[0]);
        leg1RB.AddTorque(transform.forward * torque * vectorAction[1]);
        leg2RB.AddTorque(transform.forward * torque * vectorAction[2]);

        if (body.transform.position.y < 1.1f)
        {
            AddReward(-5f);
            Done();
        }

        if (body.transform.position.x > 100f)
        {
            AddReward(5f);
            Done();
        }

        AddReward(0.1f * bodyRB.velocity.x - 0.05f * bodyRB.velocity.y);
        AddReward(body.transform.position.x - pos);

        pos = body.transform.position.x;

    }

    public override void AgentReset()
    {
        body.transform.position = body_init_pos;
        foot.transform.position = foot_init_pos;
        leg1.transform.position = leg1_init_pos;
        leg2.transform.position = leg2_init_pos;

        body.transform.rotation = body_init_rot;
        foot.transform.rotation = foot_init_rot;
        leg1.transform.rotation = leg1_init_rot;
        leg2.transform.rotation = leg2_init_rot;

        bodyRB.velocity = Vector3.zero;
        footRB.velocity = Vector3.zero;
        leg1RB.velocity = Vector3.zero;
        leg2RB.velocity = Vector3.zero;
    }

    public override void AgentOnDone()
    {

    }
}
