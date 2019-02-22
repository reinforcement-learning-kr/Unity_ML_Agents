using System;
using UnityEngine;
using System.Linq;
using MLAgents;

public class SokobanAgent : Agent
{
    [Header("Specific to GridWorld")]
    private SokobanAcademy academy;
    public float timeBetweenDecisionsAtInference;
    private float timeSinceDecision;

    [Tooltip("Selecting will turn on action masking. Note that a model trained with action " +
             "masking turned on may not behave optimally when action masking is turned off.")]
    public bool maskActions = true;

    private const int NoAction = 0;  // do nothing!
    private const int Up = 1;
    private const int Down = 2;
    private const int Left = 3;
    private const int Right = 4;

    public override void InitializeAgent()
    {
        academy = FindObjectOfType(typeof(SokobanAcademy)) as SokobanAcademy;
    }

    public override void CollectObservations()
    {
        // There are no numeric observations to collect as this environment uses visual
        // observations.

        // Mask the necessary actions if selected by the user.
        if (maskActions)
        {
            SetMask();
        }
    }

    /// <summary>
    /// Applies the mask for the agents action to disallow unnecessary actions.
    /// </summary>
    private void SetMask()
    {
        // Prevents the agent from picking an action that would make it collide with a wall
        var positionX = (int) transform.position.x;
        var positionZ = (int) transform.position.z;
        var maxPosition = academy.gridSize - 1;

        if (positionX == 0)
        {
            SetActionMask(Left);
        }

        if (positionX == maxPosition)
        {
            SetActionMask(Right);
        }

        if (positionZ == 0)
        {
            SetActionMask(Down);
        }

        if (positionZ == maxPosition)
        {
            SetActionMask(Up);
        }
    }

    // to be implemented by the developer
    public override void AgentAction(float[] vectorAction, string textAction)
    {
        AddReward(-0.01f);
        int action = Mathf.FloorToInt(vectorAction[0]);

        Vector3 targetPos = transform.position;
        switch (action)
        {
            case NoAction:
                // do nothing
                break;
            case Right:
                targetPos = transform.position + new Vector3(1f, 0, 0f);
                break;
            case Left:
                targetPos = transform.position + new Vector3(-1f, 0, 0f);
                break;
            case Up:
                targetPos = transform.position + new Vector3(0f, 0, 1f);
                break;
            case Down:
                targetPos = transform.position + new Vector3(0f, 0, -1f);
                break;
            default:
                throw new ArgumentException("Invalid action value");
        }

        Collider[] blockTest = Physics.OverlapBox(targetPos, new Vector3(0.3f, 0.3f, 0.3f));
        if (blockTest.Where(col => col.gameObject.CompareTag("wall")).ToArray().Length == 0)
        {
            if ((blockTest.Where(col => col.gameObject.CompareTag("pit")).ToArray().Length == 1) ||
                (blockTest.Where(col => col.gameObject.CompareTag("goal")).ToArray().Length == 1))
            {
                Done();
                SetReward(-1f);
            }
            else if (blockTest.Where(col => col.gameObject.CompareTag("box")).ToArray().Length == 1)
            {
                GameObject box = blockTest[0].gameObject;
                Vector3 nextBoxPos = 2 * targetPos - transform.position;
                Collider[] boxBlockTest = Physics.OverlapBox(nextBoxPos, new Vector3(0.3f, 0.3f, 0.3f));
                if (boxBlockTest.Where(col => col.gameObject.CompareTag("pit")).ToArray().Length == 1)
                {
                    Done();
                    SetReward(-1f);
                }
                else if ((boxBlockTest.Where(col => col.gameObject.CompareTag("box")).ToArray().Length == 1) ||
                    (boxBlockTest.Where(col => col.gameObject.CompareTag("wall")).ToArray().Length == 1))
                {
                    SetReward(-0.1f);
                }
                else if (boxBlockTest.Where(col => col.gameObject.CompareTag("goal")).ToArray().Length == 1)
                {
                    GameObject goal = boxBlockTest[0].gameObject;
                    transform.position = targetPos;
                    if (academy.RemoveBoxObj(box, goal) == 0) Done();
                    SetReward(1f);
                }
                else
                {
                    box.transform.position = nextBoxPos;
                    transform.position = targetPos;
                    SetReward(0.1f);
                }
            }
            else
            {
                transform.position = targetPos;
            }
        }
    }


    // to be implemented by the developer
    public override void AgentReset()
    {
        academy.AcademyReset();
    }

    public void FixedUpdate()
    {
        WaitTimeInference();
    }

    private void WaitTimeInference()
    {
        if (!academy.GetIsInference())
        {
            RequestDecision();
        }
        else
        {
            if (timeSinceDecision >= timeBetweenDecisionsAtInference)
            {
                timeSinceDecision = 0f;
                RequestDecision();
            }
            else
            {
                timeSinceDecision += Time.fixedDeltaTime;
            }
        }
    }
}
