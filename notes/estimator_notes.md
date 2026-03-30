# Estimator Notes and Questions

* encoder angle noise is incorporated in simulated input encoder measurement
but not considered in state covariance $P^-$ after prediction
	* -> add it via mapping through input jacobian $df / d\Delta\phi$ ?

* same for slip 
	* -> since we can't know slip in practice, 
  this might be an interesting parameter to learn from experiment data? What 
  about other noise parameters, especially $Q$?

* in problem yaml and robot.py line 28 the slip parameters are described as fractional/dimensionless std dev but in slip formula they are used as symmetric bounds for a uniform distribution not the std dev?

* In Dead reckoning mode we return forwarded states with additive noise to mimick mocap/IMU measurements, but we plot the estimates without noise in the visualization?

* wheel speed measurement model in OG readme is $\hat u = u_{true} + n$, 
but in the implementation we add $n$ to $\Delta\phi$, which actually makes
$\hat u = (u_{true} \cdot \Delta t + n) / \Delta t = u_{true} + n / \Delta t$ 
