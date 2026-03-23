# chaos-manifold-tarpit
# Chaos Manifold Tarpit

A proof of concept security system that uses the chaotic strange attractor to create non-linear, brute-force resistant password manifolds.

# How is works
Instead of using standard cryptographic hashing, this system uses a Lorenz system to transform a password (initial coordinates) into a 3D trajectory.

1. Strange Attractor: Using $\sigma=10, \rho=28, \beta=8/3$ to ensure aperiodic, chaotic motion.
2. Manifold Checkpoints: The system captures $(x, y, z)$ coordinates at fixed intervals, creating a "topological fingerprint" of the password.
3. Sensitivity: Due to the Butterfly Effect, a microscopic change in the input coordinate leads to a massive divergence in the final manifold.

# Goals
- Implement fixed point arithmetic for determinism
- Add tarpit for login attempts
- Use Matplotlib to visualise manifold divergence
