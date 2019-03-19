using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class BallScript : MonoBehaviour
{
    GameObject Agent;
    float ball_speed;
    float ball_random;
    void Start()
    {
    }

    public void SetBall(GameObject Agent, float ball_speed, float ball_random)
    {
        this.Agent = Agent;
        this.ball_speed = ball_speed;
        this.ball_random = ball_random;
        Random_ball();
    }

    public void Random_ball()
    {
        Vector3 vec = new Vector3(Random.Range(-4.5f, 4.5f), 0.5f, Random.Range(-4.5f, 4.5f));
        while (Vector3.Distance(Agent.transform.localPosition, vec) < 4.0f)
        {
            vec = new Vector3(Random.Range(-4.5f, 4.5f), 0.5f, Random.Range(-4.5f, 4.5f));
        }
        this.transform.localPosition = vec;
        Rigidbody rig = this.GetComponent<Rigidbody>();
        float randAngle = Mathf.Atan2((Agent.transform.localPosition.z - this.transform.localPosition.z),
            (Agent.transform.localPosition.x - this.transform.localPosition.x))
            + Random.Range(-ball_random, ball_random);
        float randSpeed = ball_speed + Random.Range(-0.5f, 0.5f);
        rig.velocity = new Vector3(randSpeed * Mathf.Cos(randAngle), 0, randSpeed * Mathf.Sin(randAngle));
    }
    // Update is called once per frame
    void Update()
    {
        Collider[] blockTest = Physics.OverlapSphere(this.transform.position,0.25f);
        if (blockTest.Where(col => col.gameObject.CompareTag("wall")).ToArray().Length != 0)
        {
            Random_ball();
        }
    }
}
