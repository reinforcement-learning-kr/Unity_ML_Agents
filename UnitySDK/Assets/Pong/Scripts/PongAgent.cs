using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class PongAgent : Agent
{
    public GameObject Ball;
    public GameObject Opponent;
    private Rigidbody RbBall;
    private Rigidbody RbAgent;
    private Rigidbody RbOpponent;
    private const int Stay = 0;
    private const int Up = 1;
    private const int Down = 2;
    Vector3 ResetPos;

    public override void InitializeAgent()
    {
        RbAgent = GetComponent<Rigidbody>();
        RbOpponent = Opponent.GetComponent<Rigidbody>();
        RbBall = Ball.GetComponent<Rigidbody>();
        ResetPos = transform.position;
    }


    public override void CollectObservations()
    {
        AddVectorObs(transform.position.x);
        AddVectorObs(transform.position.z);
        AddVectorObs(Opponent.transform.position.x);
        AddVectorObs(Opponent.transform.position.z);
        AddVectorObs(Ball.transform.position.x);
        AddVectorObs(Ball.transform.position.z);
        AddVectorObs(RbAgent.velocity.x);
        AddVectorObs(RbAgent.velocity.z);
        AddVectorObs(RbOpponent.velocity.x);
        AddVectorObs(RbOpponent.velocity.z);
        AddVectorObs(RbBall.velocity.x);
        AddVectorObs(RbBall.velocity.z);
    }


    public override void AgentAction(float[] vectorAction, string textAction)
    {
        int action = Mathf.FloorToInt(vectorAction[0]);
        switch (action)
        {
            case Stay:
                // 에이전트의 현재 좌표를 그대로 유지합니다.
                this.transform.position = this.transform.position + 0f * Vector3.right;
                break;
            case Up:
                // 에이전트의 현재 좌표를 유니티 좌표축 right 방향으로 0.3f만큼 이동시킵니다.
                this.transform.position = this.transform.position + 0.3f * Vector3.right;
                break;
            case Down:
                // 에이전트의 현재 좌표를 유니티 좌표축 left 방향으로 0.3f만큼 이동시킵니다.
                this.transform.position = this.transform.position + 0.3f * Vector3.left;
                break;
        }
    }

    public void OpponentScored()
    {
        AddReward(-1f);
        Done();
    }
    public void ScoredGoal()
    {
        AddReward(1f);
        Done();
    }
    void OnCollisionEnter(Collision col)
    {
        if (col.gameObject.CompareTag("Wall"))
        {
            RbAgent.velocity = Vector3.zero;
            RbAgent.angularVelocity = Vector3.zero;
        }
        if (col.gameObject.CompareTag("Ball"))
        {
            AddReward(0.5f);
        }
    }


    public override void AgentReset()
    {
        transform.position = ResetPos;
    }


    public override void AgentOnDone()
    {

    }
}
