# 8. Full System Architecture

Subsystems:

1. Planner → smooth reference trajectory  
2. Trajectory sampler  
3. Estimator → produces \(\hat{x},\hat{y},\hat{\theta}\)  
4. Controller  
   - geometric pose control  
   - wheel-speed PI loops  
5. Robot  
   - slip  
   - differential-drive kinematics  
6. Visualization and logging  

