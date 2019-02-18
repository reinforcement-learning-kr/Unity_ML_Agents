using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class PongGoalDetection : MonoBehaviour {

    [Header("Pong Goal Detection Setting")]
    public PongAgent AgentA;
    public PongAgent AgentB;

    private Rigidbody RbBall;

    Vector3 ResetPos;
    Vector3 velocity;

    private int agent_score1 = 0;
    private int agent_score2 = 0;

    private float max_ball_speed = 10f;
    private float min_ball_speed = 5f;

    void Start()
    {
        RbBall = GetComponent<Rigidbody>();
        ResetPos = transform.position;
    }
    public void ResetPosition()
    {
        transform.position = ResetPos;
        RbBall.velocity = Vector3.zero;
        transform.rotation = Quaternion.identity;

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

    void Update()
    {
        if (agent_score1 == 10 || agent_score2 == 10)
        {
            agent_score1 = 0;
            agent_score2 = 0;
        }

        Agent1_Score.score = agent_score1;
        Agent2_Score.score = agent_score2;
    }

    void OnCollisionEnter(Collision col)
    {
        if (col.gameObject.CompareTag("GoalA"))
        {
            ResetPosition();
            AgentB.ScoredGoal();
            AgentA.OpponentScored();
            AgentA.AgentReset();
            AgentB.AgentReset();

            agent_score2 = agent_score2 + 1;
        }
        if (col.gameObject.CompareTag("GoalB"))
        {
            ResetPosition();
            AgentA.ScoredGoal();
            AgentB.OpponentScored();
            AgentA.AgentReset();
            AgentB.AgentReset();

            agent_score1 = agent_score1 + 1;
        }
    }
}



