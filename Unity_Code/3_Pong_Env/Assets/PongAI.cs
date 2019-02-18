using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using UnityEngine.UI;
using MLAgents;
using System;

public class PongAI : MonoBehaviour {
    public GameObject ball;

	// Use this for initialization
	void Start () {
		
	}
	
	// Update is called once per frame
	void Update () {

        Vector3 pos_vector = new Vector3(ball.transform.position.x, 0.5f, -9.75f);
        this.transform.position = pos_vector;
	}

    public void OpponentScored()
    {

    }

    public void ScoredGoal()
    {

    }
}
