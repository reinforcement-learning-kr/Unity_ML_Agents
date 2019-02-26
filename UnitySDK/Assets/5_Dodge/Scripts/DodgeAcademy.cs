using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class DodgeAcademy : Academy {

    public GameObject Agent;
    public GameObject Ball;
    public GameObject Env;
    List<GameObject> balls = new List<GameObject>();

    public override void AcademyReset()
    {
        int resetSeed = (int)this.resetParameters["randomSeed"];
        if (resetSeed != 0)
            Random.seed = resetSeed;
        float ballspeed = this.resetParameters["ballSpeed"];
        int ballnum = (int)this.resetParameters["ballNum"];
        float AimRandom = this.resetParameters["ballRandom"];
        foreach(GameObject b in balls)
        {
            DestroyImmediate(b.gameObject);
        }
        balls.Clear();

        for (int i = 0; i < ballnum; i++)
        {
            GameObject b = Instantiate(Ball,Env.transform);
            BallScript script = b.GetComponent<BallScript>();
            script.SetBall(Agent, ballspeed, AimRandom);
            balls.Add(b);
        }
    }

}
