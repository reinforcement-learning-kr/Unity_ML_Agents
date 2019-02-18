using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using UnityEngine.UI;
using MLAgents;
using System;

public class PongAgent : Agent {
    [Header("Pong Agent Setting")]
    public GameObject Ball;
    public GameObject Opponent;
    public bool invertX;

    private Rigidbody RbBall;
    private Rigidbody RbAgent;
    private Rigidbody RbOpponent;

    private const int Stay = 0;
    private const int Up = 1;
    private const int Down = 2;

    Vector3 ResetPos;
    private int invertMult;


    public override void InitializeAgent()
    {
        RbAgent = GetComponent<Rigidbody>();
        RbOpponent = Opponent.GetComponent<Rigidbody>();
        RbBall = Ball.GetComponent<Rigidbody>();

        ResetPos = transform.position;

        invertMult = invertX ? 1 : -1;
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
                this.transform.position = this.transform.position + 0f * Vector3.right;
                break; 
            case Up:
                this.transform.position = this.transform.position + 0.3f * Vector3.right;
                break;
            case Down:
                this.transform.position = this.transform.position + 0.3f * Vector3.left;
                break;
            default:
                throw new ArgumentException("Invalid action value");
        }
        
    }
    
    public override void AgentReset()
    {
        transform.position = ResetPos;
        RbAgent.velocity = Vector3.zero;
        RbAgent.angularVelocity = Vector3.zero;
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
}



