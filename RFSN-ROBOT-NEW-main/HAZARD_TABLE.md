# ISO-Style Hazard Table (Auditable)

| Hazard | Cause | Detection | Mitigation |
|--------|-------|-----------|------------|
| Self collision | IK overshoot | Contact sensor | Immediate FAULT |
| Object slip | Low normal force | Wrench estimator | Abort lift |
| Runaway torque | Controller instability | Power limiter | Torque clamp |
| Workspace breach | Target error | Envelope check | Halt |
| Joint limit violation | Invalid target | Position check | Reject command |
| Excessive velocity | Trajectory error | Velocity check | Scale down |
| Power overload | High-force maneuver | Power monitor | Scale torques |
