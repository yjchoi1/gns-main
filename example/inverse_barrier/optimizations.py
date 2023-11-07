import time
import torch
from example.inverse_barrier.utils import *
from forward import rollout_with_checkpointing
from example.inverse_problem.utils import To_Torch_Model_Param


SAVE_STEP = 5

def lbfgs(
        loss_mesure,
        simulator,
        nsteps,
        mpm_inputs,
        niterations,
        checkpoint_interval,
        runout_end_true,
        centroid_true,
        n_farthest_particles,
        kinematic_positions,
        barrier_info,
        barrier_particles,
        barrier_zlocs,
        barrier_locs_true,
        optimizer,
        output_dir,
        device):

    # Variables to that saves variables in closure during optimization
    values_for_save = {}  # variables to save as outputs
    values_for_vis = {"pred": {}, "true": {}}   # variables just for visualize current optimization state

    closure_count = 0

    def closure():
        nonlocal closure_count
        start = time.time()

        closure_count += 1

        print(f"Closure: {closure_count} -----------------------------")
        print(f"True barrier locations: {barrier_locs_true}")

        optimizer.zero_grad()  # Clear previous gradients

        # Make current barrier particles with the current locations
        barrier_xlocs_true = torch.tensor(
            [barrier_locs_true[0][0], barrier_locs_true[1][0]]).to(device)
        barrier_locs = torch.stack((barrier_xlocs_true, barrier_zlocs), dim=1)
        base_height = torch.tensor(barrier_info["base_height"])
        current_barrier_particles = locate_barrier_particles(
            barrier_particles, barrier_locs, base_height)

        # Make X0 with current barrier particles
        current_initial_positions, current_particle_type, current_n_particles_per_example = get_features(
            kinematic_positions, current_barrier_particles, device)

        # GNS rollout
        predicted_positions = rollout_with_checkpointing(
            simulator=simulator,
            initial_positions=current_initial_positions,
            particle_types=current_particle_type,
            n_particles_per_example=current_n_particles_per_example,
            nsteps=nsteps,
            checkpoint_interval=checkpoint_interval,
            knwon_positions=current_initial_positions[:, 0, :]
        )

        # Get predicted position at the last timestep
        kinematic_positions_pred, stationary_positions_pred = get_positions_by_type(
            predicted_positions, current_particle_type)
        runout_end_pred = get_runout_end(
            kinematic_positions_pred[-1], n_farthest_particles)
        centroid_pred = torch.mean(kinematic_positions_pred[-1], dim=0)

        # Compute loss with before update
        if loss_mesure == "farthest_positions":
            loss = torch.mean((runout_end_pred - runout_end_true)**2)
        elif loss_mesure == "centroid":
            loss = torch.mean((centroid_pred - centroid_true)**2)
        else:
            raise ValueError("Check loss measure. Should be `farthest_positions` or `centroid`")

        print(f"loss {loss.item():.8f}")

        # Save necessary variables to visualize current optimization state
        if loss_mesure == "farthest_positions":
            values_for_vis["pred"]["kinematic_positions"] = kinematic_positions_pred  # torch.tensor
            values_for_vis["pred"]["runout_end"] = runout_end_pred  # torch.tensor
            values_for_vis["pred"]["barrier_locs"] = barrier_locs  # torch.tensor
            values_for_vis["true"]["kinematic_positions"] = kinematic_positions  # torch.tensor
            values_for_vis["true"]["runout_end"] = runout_end_true  # torch.tensor
            values_for_vis["true"]["barrier_locs"] = torch.tensor(barrier_locs_true)  # torch.tensor
        elif loss_mesure == "centroid":
            # Save necessary variables to visualize current optimization state
            values_for_vis["pred"]["kinematic_positions"] = kinematic_positions_pred  # torch.tensor
            values_for_vis["pred"]["barrier_locs"] = barrier_locs  # torch.tensor
            values_for_vis["pred"]["centroid"] = centroid_pred  # torch.tensor
            values_for_vis["true"]["kinematic_positions"] = kinematic_positions  # torch.tensor
            values_for_vis["true"]["barrier_locs"] = torch.tensor(barrier_locs_true)  # torch.tensor
            values_for_vis["true"]["centroid"] = centroid_true  # torch.tensor
        else:
            raise ValueError("Check loss measure. Should be `farthest_positions` or `centroid`")

        # Save status plot for every iteration and closure call
        visualize_state(
            vis_data=values_for_vis,
            barrier_info=barrier_info,
            mpm_inputs=mpm_inputs,
            loss=loss.item(),
            write_path=f"{output_dir}/status-e{closure_count}.png")

        # Save necessary variables to save as output
        values_for_save["current_barrier_loc"] = barrier_locs.clone().detach().cpu().numpy()
        values_for_save["predicted_positions"] = predicted_positions.clone().detach().cpu().numpy()
        values_for_save["particle_type"] = current_particle_type.clone().detach().cpu().numpy()
        values_for_save["barrier_info"] = barrier_info
        values_for_save["n_farthest_particles"] = n_farthest_particles

        # Update barrier locations
        print("Backpropagate...")
        loss.backward()
        print(barrier_zlocs.grad)
        # Print updated barrier locations
        print(f"Updated barrier locations: {barrier_locs.detach().cpu().numpy()}")

        # Save updated variables to save as output
        values_for_save["updated_barrier_loc"] = barrier_locs.clone().detach().cpu().numpy()

        # Measure time for an iteration
        end = time.time()
        time_for_iteration = end - start

        # Save optimizer state
        torch.save({
            'iteration': closure_count,
            'loss': loss.item(),
            'time_spent': time_for_iteration,
            'save_values': values_for_save,
            'updated_barrier_loc_state_dict': To_Torch_Model_Param(barrier_zlocs).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"{output_dir}/optimizer_state-e{closure_count}.pt")

        # Save animation after iteration
        if closure_count % SAVE_STEP == 0:
            render_animation(
                predicted_positions,
                current_particle_type,
                mpm_inputs,
                timestep_stride=10,
                write_path=f"{output_dir}/trj-{closure_count}.gif")

        return loss

    # Perform optimization step
    while closure_count < niterations:
        optimizer.step(closure)

        # # Enforce the boundary constraints
        # boundary_constraints = barrier_info["search_area"]
        # with torch.no_grad():  # Make sure gradients are not computed for this operation
        #     barrier_locs[:, 0].clamp_(
        #         min=boundary_constraints[0][0], max=boundary_constraints[0][1])
        #     barrier_locs[:, 1].clamp_(
        #         min=boundary_constraints[1][0], max=boundary_constraints[1][1])

    return values_for_save


# TODO (yc): change corresponding to z-location optimization
def adam(
        loss_mesure,
        simulator,
        nsteps,
        mpm_inputs,
        iteration,
        checkpoint_interval,
        runout_end_true,
        centroid_true,
        n_farthest_particles,
        kinematic_positions,
        barrier_info,
        barrier_particles,
        barrier_locs,
        barrier_locs_true,
        optimizer,
        output_dir,
        device):

    start = time.time()

    # Variables to that saves variables in closure during optimization
    values_for_save = {}  # variables to save as outputs
    values_for_vis = {"pred": {}, "true": {}}  # variables just for visualize current optimization state

    print(f"True barrier locations: {barrier_locs_true}")

    optimizer.zero_grad()  # Clear previous gradients

    # Make current barrier particles with the current locations
    base_height = torch.tensor(barrier_info["base_height"])
    current_barrier_particles = locate_barrier_particles(
        barrier_particles, barrier_locs, base_height)

    # Make X0 with current barrier particles
    current_initial_positions, current_particle_type, current_n_particles_per_example = get_features(
        kinematic_positions, current_barrier_particles, device)

    # GNS rollout
    predicted_positions = rollout_with_checkpointing(
        simulator=simulator,
        initial_positions=current_initial_positions,
        particle_types=current_particle_type,
        n_particles_per_example=current_n_particles_per_example,
        nsteps=nsteps,
        checkpoint_interval=checkpoint_interval,
        knwon_positions=current_initial_positions[:, 0, :]
    )

    # Get predicted position at the last timestep
    kinematic_positions_pred, stationary_positions_pred = get_positions_by_type(
        predicted_positions, current_particle_type)
    runout_end_pred = get_runout_end(
        kinematic_positions_pred[-1], n_farthest_particles)
    centroid_pred = torch.mean(kinematic_positions_pred[-1], dim=0)

    # Compute loss with before update
    if loss_mesure == "farthest_positions":
        loss = torch.mean((runout_end_pred - runout_end_true) ** 2)
    elif loss_mesure == "centroid":
        loss = torch.mean((centroid_pred - centroid_true) ** 2)
    else:
        raise ValueError("Check loss measure. Should be `farthest_positions` or `centroid`")

    # Save necessary variables to visualize current optimization state
    values_for_vis["pred"]["kinematic_positions"] = kinematic_positions_pred  # torch.tensor
    values_for_vis["pred"]["runout_end"] = runout_end_pred  # torch.tensor
    values_for_vis["pred"]["barrier_locs"] = barrier_locs  # torch.tensor
    values_for_vis["pred"]["centroid"] = centroid_pred  # torch.tensor
    values_for_vis["true"]["kinematic_positions"] = kinematic_positions  # torch.tensor
    values_for_vis["true"]["runout_end"] = runout_end_true  # torch.tensor
    values_for_vis["true"]["barrier_locs"] = torch.tensor(barrier_locs_true)  # torch.tensor
    values_for_vis["true"]["centroid"] = centroid_true  # torch.tensor

    # Save status plot for every iteration and closure call
    visualize_state(
        vis_data=values_for_vis,
        barrier_info=barrier_info,
        mpm_inputs=mpm_inputs,
        loss=loss.item(),
        write_path=f"{output_dir}/status-e{iteration}.png")

    # Save necessary variables to save as output
    values_for_save["current_barrier_loc"] = barrier_locs.clone().detach().cpu().numpy()
    values_for_save["predicted_positions"] = predicted_positions.clone().detach().cpu().numpy()
    values_for_save["particle_type"] = current_particle_type.clone().detach().cpu().numpy()
    values_for_save["barrier_info"] = barrier_info
    values_for_save["n_farthest_particles"] = n_farthest_particles

    # Update barrier locations
    print("Backpropagate...")
    loss.backward()
    # Print updated barrier locations
    print(f"Updated barrier locations: {barrier_locs.detach().cpu().numpy()}")

    # Save updated variables to save as output
    values_for_save["updated_barrier_loc"] = barrier_locs.clone().detach().cpu().numpy()

    # Measure time for an iteration
    end = time.time()
    time_for_iteration = end - start

    # Save optimizer state
    torch.save({
        'iteration': iteration,
        'loss': loss.item(),
        'time_spent': time_for_iteration,
        'save_values': values_for_save,
        'updated_barrier_loc_state_dict': To_Torch_Model_Param(barrier_locs).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f"{output_dir}/optimizer_state-e{iteration}.pt")

    # Perform optimization step
    optimizer.step()

    # Enforce the boundary constraints
    boundary_constraints = barrier_info["search_area"]
    with torch.no_grad():  # Make sure gradients are not computed for this operation
        barrier_locs[:, 0].clamp_(
            min=boundary_constraints[0][0], max=boundary_constraints[0][1])
        barrier_locs[:, 1].clamp_(
            min=boundary_constraints[1][0], max=boundary_constraints[1][1])

    # Save animation after iteration
    if iteration % SAVE_STEP == 0:
        render_animation(
            predicted_positions,
            current_particle_type,
            mpm_inputs,
            timestep_stride=10,
            write_path=f"{output_dir}/trj-{iteration}.gif")


    return barrier_locs, values_for_save



# def get_loss(runout_end_pred,
#              runout_end_true,
#              barrier_locs,
#              search_area,
#              penalty_scale=None):
#     """
#     Compute loss
#     Args:
#         runout_end_pred (torch.tensor):
#         runout_end_true (torch.tensor):
#         barrier_locs (torch.tensor): parameters for optimization
#         search_area (torch.tensor): barrier location constraint
#         penalty_scale (float): magnitude for penalty term
#
#     Returns:
#         loss
#
#     """
#     if penalty_scale is None:
#         loss = torch.mean((runout_end_pred - runout_end_true) ** 2)
#     else:
#         penalty = penalty_scale *
#         loss = torch.mean((runout_end_pred - runout_end_true) ** 2)
#
#     return loss