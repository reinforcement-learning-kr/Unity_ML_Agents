using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class body_dist : MonoBehaviour {

    public Camera cam;

	// Use this for initialization
	void Start () {
		
	}
	
	// Update is called once per frame
	void Update () {

        DistanceText.dist = this.transform.position.x;
        cam.transform.position = new Vector3(this.transform.position.x, cam.transform.position.y, cam.transform.position.z);
    }
}
