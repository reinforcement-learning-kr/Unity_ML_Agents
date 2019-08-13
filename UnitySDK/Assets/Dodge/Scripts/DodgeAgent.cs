using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using MLAgents;

public class DodgeAgent : Agent {

    DodgeAcademy academy;
    Rigidbody rigidbody;
    public float speed = 30f;

    public override void InitializeAgent()
    {
        base.InitializeAgent();
        academy = FindObjectOfType(typeof(DodgeAcademy)) as DodgeAcademy;
        rigidbody = GetComponent<Rigidbody>();
    }

    public override void CollectObservations()
    {
        RaycastHit hit;
        Ray ray;
        float Angle;
        float raycount = 30f;
        List<Vector3> debugRay = new List<Vector3>();
        for(float i =0; i < raycount; i++)
        {
            Angle = i * 2.0f * Mathf.PI / raycount;
            ray = new Ray(this.transform.position, new Vector3(Mathf.Cos(Angle), 0, Mathf.Sin(Angle)));

            if(Physics.Raycast(ray,out hit))
            {
                AddVectorObs(hit.distance);
                debugRay.Add(hit.point);
            }
        }
        //debug ray visualize
        for (int i = 0; i < debugRay.Count - 1; i++)
            Debug.DrawRay(debugRay[i], debugRay[i + 1] - debugRay[i], Color.green);
        Debug.DrawRay(debugRay[debugRay.Count - 1], debugRay[0] - debugRay[debugRay.Count - 1], Color.green);
    }

    public override void AgentAction(float[] vectorAction, string textAction)
	{
        Vector3 Force = new Vector3();
        switch(vectorAction[0])
        {
            case 1:
                Force = new Vector3(-1, 0, 0) * speed;
                break;
            case 2:
                Force = new Vector3(0, 0, 1) * speed;
                break;
            case 3:
                Force = new Vector3(0, 0, -1) * speed;
                break;
            case 4:
                Force = new Vector3(1, 0, 0) * speed;
                break;
            default:
                Force = new Vector3(0, 0, 0);
                break;
        }
        rigidbody.AddForce(Force);

        Collider[] blockTest = Physics.OverlapBox(this.transform.position, new Vector3(0.26f, 0.26f, 0.26f));
        if (blockTest.Where(col => col.gameObject.CompareTag("ball")).ToArray().Length != 0)
        {
            Done();
            SetReward(-1f);
        }
        else
        {
            SetReward(0.1f);
        }
    }

    public override void AgentReset()
    {
        speed = academy.resetParameters["agentSpeed"];
        this.transform.localPosition = new Vector3(0, 0.5f, 0);
        rigidbody.velocity = new Vector3(0, 0, 0);
        academy.AcademyReset();
    }
}
