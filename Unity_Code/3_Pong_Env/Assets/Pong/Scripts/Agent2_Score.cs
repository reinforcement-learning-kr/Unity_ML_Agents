using UnityEngine;
using UnityEngine.UI;
using System.Collections;

public class Agent2_Score : MonoBehaviour
{
    public static float score;
    Text text;
    void Awake()
    {
        text = GetComponent<Text>();
        score = 0;
    }
    void Update()
    {
        text.text = score.ToString("0");
    }
}