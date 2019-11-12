using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PongGoalDetection : MonoBehaviour
{
    public PongAgent AgentA;
    public PongAgent AgentB;
    private Rigidbody RbBall;
    Vector3 ResetPos;
    Vector3 velocity;
    private float max_ball_speed = 10f;
    private float min_ball_speed = 5f;

    void Start()
    {
        RbBall = GetComponent<Rigidbody>();
        ResetPos = transform.position;
        ResetPosition();
    }

    public void ResetPosition()
    {
        transform.position = ResetPos;
        RbBall.velocity = Vector3.zero;
        transform.rotation = Quaternion.identity;

        float rand_num = Random.Range(-1f, 1f);

        if (rand_num < -0.5f)
        {
            // 오른쪽 위로 공을 움직입니다.
            velocity = new Vector3(Random.Range(min_ball_speed, max_ball_speed), 0, Random.Range(min_ball_speed, max_ball_speed));
        }
        else if (rand_num < 0f)
        {
            // 오른쪽 아래로 공을 움직입니다.
            velocity = new Vector3(Random.Range(min_ball_speed, max_ball_speed), 0, Random.Range(-max_ball_speed, -min_ball_speed));
        }
        else if (rand_num < 0.5f)
        {
            // 왼쪽 위로 공을 움직입니다.
            velocity = new Vector3(Random.Range(-max_ball_speed, -min_ball_speed), 0, Random.Range(min_ball_speed, max_ball_speed));
        }
        else
        {
            // 왼쪽 아래로 공을 움직입니다.
            velocity = new Vector3(Random.Range(-max_ball_speed, -min_ball_speed), 0, Random.Range(-max_ball_speed, -min_ball_speed));
        }
        RbBall.AddForce(velocity);
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
        }
        if (col.gameObject.CompareTag("GoalB"))
        {
            ResetPosition();
            AgentA.ScoredGoal();
            AgentB.OpponentScored();
            AgentA.AgentReset();
            AgentB.AgentReset();
        }
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
