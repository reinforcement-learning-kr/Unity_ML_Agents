using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class PongAcademy : Academy {

    [Header("Pong Setting")]
    public GameObject AgentA;
    public GameObject AgentB;
    public GameObject Ball;

    Vector3 ResetPosBall;
    Vector3 ResetPosAgentA;
    Vector3 ResetPosAgentB;
    Vector3 velocity;

    Rigidbody RbBall;
    Rigidbody RbAgentA;
    Rigidbody RbAgentB;

    private float max_ball_speed = 10f;
    private float min_ball_speed = 7f;

    public override void InitializeAcademy()
    {
        ResetPosBall = Ball.transform.position;
        ResetPosAgentA = AgentA.transform.position;
        ResetPosAgentB = AgentB.transform.position;

        RbBall = Ball.GetComponent<Rigidbody>();
        RbAgentA = AgentA.GetComponent<Rigidbody>();
        RbAgentB = AgentB.GetComponent<Rigidbody>();

        float rand_num = Random.Range(-1f, 1f);

        if (rand_num < -0.5f)
        {
            velocity = new Vector3(Random.Range(min_ball_speed, max_ball_speed), 0, Random.Range(min_ball_speed, max_ball_speed));
        }
        else if (rand_num < 0f)
        {
            velocity = new Vector3(Random.Range(min_ball_speed, max_ball_speed), 0, Random.Range(-max_ball_speed, -min_ball_speed));
        }
        else if (rand_num < 0.5f)
        {
            velocity = new Vector3(Random.Range(-max_ball_speed, -min_ball_speed), 0, Random.Range(min_ball_speed, max_ball_speed));
        }
        else
        {
            velocity = new Vector3(Random.Range(-max_ball_speed, -min_ball_speed), 0, Random.Range(-max_ball_speed, -min_ball_speed));
        }

        RbBall.AddForce(velocity);
    }

    public override void AcademyReset()
    {
        Ball.transform.position = ResetPosBall;
        AgentA.transform.position = ResetPosAgentA;
        AgentB.transform.position = ResetPosAgentB;

        RbBall.velocity = Vector3.zero;
        Ball.transform.rotation = Quaternion.identity;
        RbAgentA.velocity = Vector3.zero;
        RbAgentA.angularVelocity = Vector3.zero;
        RbAgentB.velocity = Vector3.zero;
        RbAgentB.angularVelocity = Vector3.zero;

        float rand_num = Random.Range(-1f, 1f);

        if (rand_num < -0.5f)
        {
            velocity = new Vector3(Random.Range(min_ball_speed, max_ball_speed), 0, Random.Range(min_ball_speed, max_ball_speed));
        }
        else if (rand_num < 0f)
        {
            velocity = new Vector3(Random.Range(min_ball_speed, max_ball_speed), 0, Random.Range(-max_ball_speed, -min_ball_speed));
        }
        else if (rand_num < 0.5f)
        {
            velocity = new Vector3(Random.Range(-max_ball_speed, -min_ball_speed), 0, Random.Range(min_ball_speed, max_ball_speed));
        }
        else
        {
            velocity = new Vector3(Random.Range(-max_ball_speed, -min_ball_speed), 0, Random.Range(-max_ball_speed, -min_ball_speed));
        }

        RbBall.AddForce(velocity);
    }

}


