using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class PongAcademy : Academy
{
    public GameObject Ball; 

    public override void AcademyReset()
    {
        Ball.GetComponent<PongGoalDetection>().ResetPosition();
    }

    public override void AcademyStep()
    {

    }
}
