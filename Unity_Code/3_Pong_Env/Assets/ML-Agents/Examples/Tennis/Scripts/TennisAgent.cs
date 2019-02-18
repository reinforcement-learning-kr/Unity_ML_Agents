using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using MLAgents;

public class TennisAgent : Agent
{
    [Header("Specific to Tennis")]
    public GameObject ball;
    public bool invertX;
    public int score;
    public GameObject scoreText;
    public GameObject myArea;
    public GameObject opponent;

    private Text textComponent;
    private Rigidbody agentRb;
    private Rigidbody ballRb;
    private float invertMult;

    public override void InitializeAgent()
    {
        agentRb = GetComponent<Rigidbody>();
        ballRb = GetComponent<Rigidbody>();
        textComponent = scoreText.GetComponent<Text>();
    }

    public override void CollectObservations()
    {
        // Agent의 위치에 대한 정보
        AddVectorObs(invertMult * (transform.position.x - myArea.transform.position.x));
        AddVectorObs(transform.position.z - myArea.transform.position.z);

        // X 축 & Z축 이동 속도 정보 
        AddVectorObs(invertMult * agentRb.velocity.x);
        AddVectorObs(invertMult * agentRb.velocity.z);
        
        // Ball과 Agent와의 거리에 대한 observation X축, y 축
        AddVectorObs(invertMult * (ball.transform.position.x - myArea.transform.position.x));
        AddVectorObs(ball.transform.position.z - myArea.transform.position.z);
        // Ball의 속도에 대한 정보 
        AddVectorObs(invertMult * ballRb.velocity.x);
        AddVectorObs(ballRb.velocity.z);
    }


    public override void AgentAction(float[] vectorAction, string textAction)
    {
        var moveX = Mathf.Clamp(vectorAction[0], -1f, 1f) * invertMult;
        var moveY = Mathf.Clamp(vectorAction[1], -1f, 1f);
        
        if (moveY > 0.5 && transform.position.y - transform.parent.transform.position.y < -1.5f)
        {
            agentRb.velocity = new Vector3(agentRb.velocity.x, 7f, 0f);
        }

        agentRb.velocity = new Vector3(moveX * 30f, agentRb.velocity.y, 0f);

        if (invertX && transform.position.x - transform.parent.transform.position.x < -invertMult || 
            !invertX && transform.position.x - transform.parent.transform.position.x > -invertMult)
        {
                transform.position = new Vector3(-invertMult + transform.parent.transform.position.x, 
                                                            transform.position.y, 
                                                            transform.position.z);
        }

        textComponent.text = score.ToString();
    }

    public override void AgentReset()
    {
        invertMult = invertX ? -1f : 1f;

        transform.position = new Vector3(-invertMult * Random.Range(6f, 8f), -1.5f, 0f) + transform.parent.transform.position;
        agentRb.velocity = new Vector3(0f, 0f, 0f);
    }
}
